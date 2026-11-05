#!/bin/bash
# Stella throughput sweep: rewrite config.pbtxt, hot-reload, measure, repeat.
#
# RUN THIS ON THE TRITON HOST, not from the laptop:
#   scp sweep.sh loadgen.py 7700x:/tmp/
#   ssh 7700x 'docker cp /tmp/loadgen.py triton-stella:/tmp/ && bash /tmp/sweep.sh 20'
#
# Each row rewrites the whole config rather than sed-ing it. config.pbtxt puts
# key: and value: on separate lines, so patterns written against
# `KEY { ... }` match nothing and the row silently measures the previous
# config. Rewriting the file removes that failure mode entirely.
#
# After each rewrite the model is unloaded and reloaded through the repository
# API. That needs --model-control-mode=explicit on the server, which
# run_triton_stella.sh already sets. Reload costs ~25s (1.7 GB of weights), so
# the sweep is dominated by setup, not measurement.
#
# Batch size is read from Triton's own counters, not inferred from the config:
# requests/exec_count is how many requests the scheduler actually grouped per
# execute() call. That distinction is the whole point -- dynamic batching can
# be enabled and still never fire at low concurrency.
#
# Requires: server up via run_triton_stella.sh --bg, loadgen.py at
# /tmp/loadgen.py inside the container.
#
# Rows are "label:batching:count:concurrency", space separated. Override with
# the ROWS env var to explore a different corner of the space:
#
#   ROWS='deep:on:4:64' ./sweep.sh 20
#
#   ./sweep.sh [seconds-per-row]

set -e
DUR=${1:-20}
CONTAINER=${CONTAINER:-triton-stella}
MODEL=stella400m
CFG=/AI/models/triton/$MODEL/config.pbtxt
ENV_PATH=/opt/ml/model/miniLM/minilm_env.tar.gz

write_config() {
  # $1 = batching ("on"|"off"), $2 = instance count
  local batch=$1 count=$2
  cat > $CFG <<EOF
name: "$MODEL"
backend: "python"
max_batch_size: 128
EOF

  if [ "$batch" = "on" ]; then
    cat >> $CFG <<'EOF'

dynamic_batching {
  preferred_batch_size: [ 2, 4, 8, 16, 32, 64, 128 ]
  max_queue_delay_microseconds: 5000
}
EOF
  fi

  cat >> $CFG <<EOF

instance_group [
  {
    count: $count
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

# write_config emits a comment-free config, so stash the annotated one first and
# put it back at the end. Otherwise a sweep silently deletes the documentation
# from the live config and the next reader wonders where the knob notes went.
if [ ! -f "$CFG.orig" ]; then
  cp "$CFG" "$CFG.orig"
fi

reload_model() {
  curl -s -X POST http://localhost:8000/v2/repository/models/$MODEL/unload >/dev/null
  sleep 2
  curl -s -X POST http://localhost:8000/v2/repository/models/$MODEL/load >/dev/null
  for i in $(seq 1 90); do
    if curl -s -o /dev/null http://localhost:8000/v2/models/$MODEL/ready; then return 0; fi
    sleep 1
  done
  echo "  !! reload timed out"; return 1
}

snap() {
  curl -s http://localhost:8002/metrics | awk '
    /^nv_inference_request_success\{.*model="stella400m"/ {r+=$2}
    /^nv_inference_exec_count\{.*model="stella400m"/      {e+=$2}
    END {print r+0, e+0}'
}

run_row() {
  local label=$1 batch=$2 count=$3 conc=$4
  echo
  echo "=== $label  (batching=$batch count=$count conc=$conc dur=${DUR}s) ==="
  write_config "$batch" "$count"
  reload_model

  read -r R0 E0 <<<"$(snap)"
  nvidia-smi dmon -s um -d 1 -c $((DUR + 2)) > /tmp/dmon.txt 2>&1 &
  DPID=$!
  sleep 1
  docker exec $CONTAINER python3 /tmp/loadgen.py "$conc" "$DUR"
  wait $DPID
  read -r R1 E1 <<<"$(snap)"

  REQ=$((R1 - R0)); EXEC=$((E1 - E0))
  echo "  METRICS requests=$REQ execs=$EXEC req_per_exec=$(awk -v a=$REQ -v b=$EXEC 'BEGIN{if(b>0)printf "%.2f", a/b; else print "n/a"}')"
  # dmon columns: gpu sm mem enc dec jpg ofa fb bar1 ccpm. $2+0 forces numeric
  # comparison; without it awk compares strings and the max is wrong.
  awk 'BEGIN{msm=0;mmem=0;mfb=0;n=0;ssm=0;smem=0}
       $1=="0" {sm=$2+0; mem=$3+0; fb=$8+0;
         if(sm>msm)msm=sm; ssm+=sm;
         if(mem>mmem)mmem=mem; smem+=mem;
         if(fb>mfb)mfb=fb; n++}
       END {if(n) printf "  GPU sm avg=%.0f%% max=%d%% | mem.avg=%.0f%% max=%d%% | fb.max=%dMB | samples=%d\n", ssm/n, msm, smem/n, mmem, mfb, n}' /tmp/dmon.txt
}

ROWS=${ROWS:-'baseline_no_batching:off:1:1 dynamic_batching_on:on:1:16 instance_count_2:on:2:16 instance_count_4:on:4:16 instance_count_4_deep:on:4:64'}

echo "stella400m sweep  $(date -Is)  dur=${DUR}s/row"
echo "host: $(hostname)  container: $CONTAINER"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

for spec in $ROWS; do
  IFS=: read -r label batch count conc <<<"$spec"
  run_row "$label" "$batch" "$count" "$conc"
done

echo
echo "=== restoring original config ==="
cp "$CFG.orig" "$CFG"
reload_model
echo "sweep complete"
