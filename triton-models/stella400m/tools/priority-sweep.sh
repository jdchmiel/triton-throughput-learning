#!/bin/bash
# Co-tenancy sweep: hold a 100 TPS search SLO while indexing load climbs.
#
# RUN THIS ON THE TRITON HOST, not from the laptop:
#   scp priority-sweep.sh priorityload.py 7700x:/tmp/
#   ssh 7700x 'docker cp /tmp/priorityload.py triton-stella:/tmp/ && bash /tmp/priority-sweep.sh'
#
# The question is whether a low-priority indexing queue can soak up spare
# capacity without touching the search SLO. Answering it needs three things
# the earlier sweep did not do:
#
#  1. OPEN-LOOP load. Fixed offered rate per class, not fixed worker counts.
#     Closed-loop would make indexing win by construction and the graph would
#     measure the harness.
#  2. A CONTROL. The "no priority" mode sends no priority header at all, so
#     both classes land in default_priority_level and compete FIFO. If the
#     priority curve does not beat that curve, priority is doing nothing and
#     the graph is decoration.
#  3. A CLIENT-CEILING GUARD. pool_wait_max says whether the generator could
#     afford the offered rate. Without it, a saturated client looks exactly
#     like a saturated server -- which is the mistake this campaign has
#     already made once.
#
# Indexing load is ramped INSIDE a single run so the graph is one continuous
# line. The steps are fractions of indexing capacity measured at the start of
# this run, because 512-token passages cost far more than the short texts the
# earlier sweep used; an absolute ramp picked from that table would either
# never threaten the SLO or blow straight through it.

set -e
CONTAINER=${CONTAINER:-triton-stella}
MODEL=stella400m
CFG=/AI/models/triton/$MODEL/config.pbtxt
ENV_PATH=/opt/ml/model/miniLM/minilm_env.tar.gz
COUNT=${COUNT:-4}
SEARCH_RATE=${SEARCH_RATE:-100}
SLO_MS=${SLO_MS:-100}
# Bucket width is tied to the step so every reported percentile sits inside a
# single load step; a bucket straddling a step change blends two loads.
STEP=${STEP:-15}
STEPS=${STEPS:-'0 0.1 0.25 0.5 0.75 1.0 1.5'}
OUT=${OUT:-/tmp/priority-sweep.jsonl}
# 256 segfaults Triton (exit 139) partway through model load; 128 loads fine.
# Stay under it and keep the two classes' in-flight caps summing below it, or
# queueing moves into the socket backlog and looks like model latency.
HTTP_THREADS=${HTTP_THREADS:-128}
# Search gets the bigger share. If the cap throttles search we would see a
# false SLO break caused by the client rather than the server, so it is sized
# to tolerate ~480ms of search latency at 100 TPS -- well past the point where
# the SLO is already broken, so the cap never bites first.
SEARCH_INFLIGHT=${SEARCH_INFLIGHT:-48}
# Leave 8 threads of slack: an in-flight request occupies an HTTP worker for
# its whole round trip, so the two classes' caps must sum below HTTP_THREADS or
# queueing hides in the socket backlog and reads as model latency.
INDEX_INFLIGHT=${INDEX_INFLIGHT:-$((HTTP_THREADS - SEARCH_INFLIGHT - 8))}
# Calibration is deliberately NOT bound by INDEX_INFLIGHT: a cap there would
# understate capacity and shrink the whole ramp. It is bound by the thread
# ceiling anyway, and cap_wait in the calibration summary says whether even
# that bound was reached.
TABLE=${TABLE:-/tmp/priority-sweep.md}
# Priority is a QUEUE policy: it reorders requests waiting behind a batch that
# is already running. So it can only help when in-flight exceeds one batch.
# The two caps above sum to ~120, which is why this must not stay at 128 -- at
# max_batch_size 128 every arrival fits inside the batch currently being
# formed, the queue is always empty, and priority is provably a no-op. Measured:
# 609ms vs 596ms (no effect) at 128; 392ms vs 1741ms at 16. Smaller batches
# also cost throughput, so this is a real trade and both values are worth
# running; 16 is the default because it is the one where the mechanism can act.
MAX_BATCH=${MAX_BATCH:-16}

write_config() {
  # $1 = priority ("on"|"off"). Whole-file rewrite, not sed: config.pbtxt puts
  # key: and value: on separate lines, so a sed pattern written against
  # `KEY { ... }` matches nothing and the row silently measures the previous
  # config.
  local prio=$1
  # Two config choices decide whether this experiment can see anything at all.
  #
  # 1. preferred_batch_size entries must all be <= max_batch_size or Triton
  #    refuses the config outright ("dynamic batching preferred size must be <=
  #    max batch size") and the model silently stays unloaded. Every request
  #    then errors, calibration reports 0 rps, and the ramp collapses to all
  #    zeros, which reads like a capacity finding instead of a typo.
  # 2. Listing *small* preferred sizes (2, 4, 8, ...) makes the batcher fire a
  #    batch the moment that many requests are waiting. The queue is then
  #    permanently near-empty and there is nothing for priority_levels to
  #    reorder -- the guard measures ratio ~1.0 and correctly reports priority
  #    as dead, but for the wrong reason. Requesting only the full batch makes
  #    the batcher hold requests until it can fill MAX_BATCH, which is what
  #    builds the queue depth priority scheduling acts on.
  cat > $CFG <<EOF
name: "$MODEL"
backend: "python"
max_batch_size: $MAX_BATCH
EOF

  cat >> $CFG <<EOF

dynamic_batching {
  preferred_batch_size: [ $MAX_BATCH ]
  max_queue_delay_microseconds: 5000
EOF

  if [ "$prio" = "on" ]; then
    # Levels start at 1 and 1 is the HIGHEST priority. Search is sent at 1 and
    # indexing at 2; anything that arrives without a header falls to
    # default_priority_level, which is deliberately the low one so a caller
    # that knows nothing about priorities cannot outrank search.
    cat >> $CFG <<'EOF'

  priority_levels: 2
  default_priority_level: 2
}
EOF
  else
    echo "}" >> $CFG
  fi

  cat >> $CFG <<EOF

instance_group [
  {
    count: $COUNT
    kind: KIND_GPU
  }
]

parameters: {
  key: "EXECUTION_ENV_PATH"
  value: { string_value: "$ENV_PATH" }
}
parameters: {
  key: "DTYPE"
  value: { string_value: "float16" }
}
parameters: {
  key: "PROMPT_MODE"
  value: { string_value: "always" }
}
parameters: {
  key: "MAX_LENGTH"
  value: { string_value: "512" }
}
parameters: {
  key: "TORCH_THREADS"
  value: { string_value: "0" }
}

input [
  {
    name: "TEXT"
    data_type: TYPE_STRING
    dims: [ 1 ]
  },
  {
    name: "USE_PROMPT"
    data_type: TYPE_BOOL
    dims: [ 1 ]
    optional: true
  }
]

output [
  {
    name: "EMBEDDING"
    data_type: TYPE_FP32
    dims: [ 1024 ]
  }
]
EOF
}

reload_model() {
  # Expected batch size is read from the config we just wrote rather than from
  # MAX_BATCH, so the restore path (which loads the pristine config, whose
  # batch size is deliberately not MAX_BATCH) is verified too instead of being
  # reported as a stale config on a perfectly successful restore.
  local want
  want=$(awk '/^max_batch_size:/ {print $2; exit}' "$CFG")
  curl -s -X POST http://localhost:8000/v2/repository/models/$MODEL/unload >/dev/null
  sleep 2
  local load
  load=$(curl -s -X POST http://localhost:8000/v2/repository/models/$MODEL/load)
  # A rejected config does NOT make this call fail loudly -- Triton answers 200
  # and leaves the model unloaded. Every request then errors, calibration
  # reports 0 rps, and the ramp collapses to all zeros, which reads like a
  # capacity finding instead of a typo in the config. Surface it here.
  if [ -n "$load" ]; then
    echo "  !! load rejected: $load"
    return 1
  fi
  for i in $(seq 1 120); do
    if curl -s -o /dev/null http://localhost:8000/v2/models/$MODEL/ready; then
      # Confirm the server is running the config we just wrote, not the last
      # one it successfully loaded.
      local got
      got=$(curl -s http://localhost:8000/v2/models/$MODEL/config | \
            python3 -c 'import sys,json;print(json.load(sys.stdin).get("max_batch_size"))' 2>/dev/null)
      if [ "$got" != "$want" ]; then
        echo "  !! max_batch_size is $got, expected $want -- stale config"
        return 1
      fi
      return 0
    fi
    sleep 1
  done
  echo "  !! reload timed out"; return 1
}

# run_mode <priority on|off> <label>
# Is priority scheduling actually in effect?
#
# The obvious guard -- send a level outside [1, priority_levels] and require a
# rejection -- does not work: Triton CLAMPS an out-of-range level rather than
# refusing the request, so priority 99 returns HTTP 200 with priority_levels: 2
# live. Written that way the guard aborts a healthy server, which is worse than
# no guard at all.
#
# So the check is behavioural, and it lives in priority-probe.py: saturate with
# low-priority traffic, then compare a high-priority probe against a probe that
# carries no priority. That also catches the expensive failure this campaign
# already paid for once -- priority sent as an HTTP header is accepted and
# silently ignored, so both arms run identically and the result reads as
# "priority does nothing" rather than "priority was never delivered".
assert_priority_live() {
  local out rc
  out=$(docker exec $CONTAINER python3 /tmp/priority-probe.py \
          --senders $((MAX_BATCH * 6)) --settle 5 --probes 6 2>&1)
  rc=$?
  echo "   priority probe: $out"
  if [ $rc -ne 0 ]; then
    echo "   !! priority NOT in effect. Do not run the sweep -- both arms would"
    echo "   !! measure the same scheduler. Check that priority-probe.py is in"
    echo "   !! the container and that priority travels in the request body."
    return 1
  fi
  echo "   priority mechanism live"
}

run_mode() {
  local prio=$1 label=$2
  echo
  echo "=== $label  (priority=$prio count=$COUNT search=$SEARCH_RATE) ==="
  write_config "$prio"
  reload_model
  if [ "$prio" = "on" ]; then
    assert_priority_live || return 1
  fi

  # Calibration: how much indexing can this config do on its own? Everything
  # downstream is expressed as a fraction of this, so it has to be measured
  # with search absent -- otherwise the ramp is scaled by a number that
  # already includes search stealing capacity.
  echo "-- calibrating indexing capacity (indexing only, over-offered)"
  local cal
  cal=$(docker exec $CONTAINER python3 /tmp/priorityload.py \
      --search-rate 0 --index-rate 200 --indexing-max-inflight $((HTTP_THREADS - 8)) \
      --duration 12 --warmup 2 --label "${label}-cal" \
      $([ "$prio" = "on" ] && echo "" || echo "--no-priority") 2>&1 >/dev/null)
  local cap
  cap=$(echo "$cal" | python3 -c 'import sys,json;print(json.load(sys.stdin)["indexing_delivered"])' 2>/dev/null || echo 0)
  echo "   indexing capacity alone: ${cap} rps"
  echo "$cal" >> $OUT
  # A zero here means the model was not serving, not that it has no capacity.
  # Every downstream step is a fraction of this number, so a zero would produce
  # a perfectly flat, perfectly meaningless graph.
  if [ "$(awk -v c="$cap" 'BEGIN{print (c+0 < 1) ? 1 : 0}')" = "1" ]; then
    echo "  !! calibration delivered <1 rps -- model is not serving, aborting $label"
    echo "$cal" | python3 -c 'import sys,json;d=json.load(sys.stdin);print("  !! errors:",d.get("indexing_errors"),"search errors:",d.get("search_errors"))' 2>/dev/null || true
    return 1
  fi

  # Build the absolute ramp from the fractions.
  local ramp="" frac
  for frac in $STEPS; do
    local r
    r=$(awk -v c="$cap" -v f="$frac" 'BEGIN{printf "%.0f", c*f}')
    [ -z "$ramp" ] && ramp="$r" || ramp="$ramp,$r"
  done
  echo "-- ramp (offered indexing rps): $ramp  step=${STEP}s"

  local nsteps dur
  nsteps=$(echo "$STEPS" | wc -w)
  dur=$((nsteps * STEP))
  # dmon samples the whole row: calibration, the ramp, and reload slack.
  nvidia-smi dmon -s um -d 1 -c $((dur + 90)) > /tmp/dmon-$label.txt 2>&1 &
  local dpid=$!
  sleep 1
  docker exec $CONTAINER python3 /tmp/priorityload.py \
      --search-rate $SEARCH_RATE \
      --search-max-inflight $SEARCH_INFLIGHT --indexing-max-inflight $INDEX_INFLIGHT \
      --index-ramp "$ramp" --ramp-step $STEP \
      --duration $dur --bucket-seconds $STEP \
      --warmup 3 --label "$label" \
      $([ "$prio" = "on" ] && echo "" || echo "--no-priority") \
      >> $OUT 2>/tmp/summary-$label.txt
  wait $dpid 2>/dev/null || true
  cat /tmp/summary-$label.txt >> $OUT

  OUT=$OUT TABLE=$TABLE DMON=/tmp/dmon-$label.txt python3 - "$label" "$prio" "$cap" "$SLO_MS" <<'PYEOF'
import json, os, sys
label, prio, cap, slo = sys.argv[1:5]
dmon = os.environ["DMON"]
rows = [json.loads(l) for l in open(os.environ["OUT"]) if l.strip()]
per = [r for r in rows if r.get("label") == label and r.get("class") in ("search", "indexing")]
summ = [r for r in rows if r.get("label") == label and r.get("class") == "summary"]
if not summ:
    print("   !! no summary for", label); sys.exit(0)
s = max(summ, key=lambda r: r.get("duration", 0))
bucket = s.get("bucket_seconds", 5.0)
# Drop the first bucket: it straddles warmup, and a cold-cache percentile there
# would read as an SLO break that has nothing to do with co-tenancy.
sr = sorted(r["p99"] for r in per if r["class"] == "search" and r["t"] >= bucket)
if not sr:
    print("   !! no search buckets for", label); sys.exit(0)
# Stability and violation are different questions. Median-of-bucket-p99 is how
# the service behaved; max is whether it ever broke the SLO. Reporting only the
# max makes a stable service look fragile; only the median hides a real breach.
med = sr[len(sr) // 2]
worst = max(sr)
ok = "PASS" if worst <= float(slo) else "BREAK"
offered = max((r.get("offered_rate", 0) for r in per if r["class"] == "indexing"), default=0)
sm = fb = 0
try:
    for line in open(dmon):
        f = line.split()
        if len(f) > 7 and f[0] == "0":
            sm = max(sm, int(f[2])); fb = max(fb, int(f[7]))
except FileNotFoundError:
    pass
print(f"   {label}: search {s['search_delivered']} rps "
      f"| p99 median {med:.1f}ms max {worst:.1f}ms {ok} "
      f"| indexing offered peak {offered:.0f} delivered {s['indexing_delivered']} "
      f"| guard pool_wait search={s['search_pool_wait_max']}ms "
      f"index={s['indexing_pool_wait_max']}ms "
      f"| GPU sm={sm}% fb={fb}MB")
with open(os.environ["TABLE"], "a") as fh:
    fh.write(f"| {label} | {prio} | {cap} | {s['search_delivered']} | "
             f"{med:.1f} | {worst:.1f} | {ok} | {offered:.0f} | "
             f"{s['indexing_delivered']} | {s['indexing_p99']} | "
             f"{s['search_pool_wait_max']} | {s['indexing_pool_wait_max']} | "
             f"{sm} | {fb} |\n")
PYEOF
}

if [ ! -f "$CFG.orig" ]; then cp "$CFG" "$CFG.orig"; fi
: > $OUT
echo "| mode | priority | index cap rps | search rps | search p99 median | search p99 max | SLO | index offered peak | index delivered | index p99 | pool wait s | pool wait i | GPU sm% | GPU MB |" > $TABLE
echo "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|" >> $TABLE

echo "stella400m co-tenancy sweep  $(date -Is)"
echo "host: $(hostname)  container: $CONTAINER"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Control first: if priority cannot beat FIFO, that is the finding.
run_mode off no_priority
run_mode on priority

echo
echo "=== restoring original config ==="
cp "$CFG.orig" "$CFG"
reload_model
echo "sweep complete"
echo "jsonl: $OUT"
echo "table: $TABLE"
cat $TABLE
