#!/bin/bash
# Measure intent-replica: throughput, latency, GPU utilisation and the real
# dynamic-batch size, all in one pass.
#
#   ./measure.sh <concurrency> [seconds]
#
# Batch size is read from Triton's own counters rather than inferred from
# config.pbtxt: requests/exec_count is how many requests the scheduler
# actually grouped per execute() call. That distinction matters here, because
# the scheduler batches even when the model code loops over requests one at a
# time, in which case the batching buys nothing.
#
# Expects loadgen.py at /tmp/loadgen.py inside the container named "triton".
set -e
CONC=${1:?usage: measure.sh <concurrency> [seconds]}
DUR=${2:-20}

snap() {
  curl -s http://localhost:8002/metrics | awk '
    /^nv_inference_request_success\{gpu_uuid.*model="intent-replica"/ {r+=$2}
    /^nv_inference_exec_count\{gpu_uuid.*model="intent-replica"/      {e+=$2}
    END {print r+0, e+0}'
}

read -r R0 E0 <<<"$(snap)"
nvidia-smi dmon -s um -d 1 -c $((DUR + 2)) > /tmp/dmon.txt 2>&1 &
DPID=$!
sleep 1
docker exec triton python3 /tmp/loadgen.py "$CONC" "$DUR"
wait $DPID
read -r R1 E1 <<<"$(snap)"

REQ=$((R1 - R0)); EXEC=$((E1 - E0))
echo "METRICS requests=$REQ execs=$EXEC req_per_exec=$(echo "scale=2; $REQ/$EXEC" | bc)"
# dmon columns: gpu sm mem enc dec jpg ofa fb bar1 ccpm
awk 'BEGIN{msm=0;mmem=0;mfb=0;n=0;ssm=0;smem=0}
     $1=="0" {sm=$2+0; mem=$3+0; fb=$8+0;
       if(sm>msm)msm=sm; ssm+=sm;
       if(mem>mmem)mmem=mem; smem+=mem;
       if(fb>mfb)mfb=fb; n++}
     END {if(n) printf "GPU sm avg=%.0f%% max=%d%% | mem.avg=%.0f%% max=%d%% | fb.max=%dMB | samples=%d\n", ssm/n, msm, smem/n, mmem, mfb, n}' /tmp/dmon.txt
