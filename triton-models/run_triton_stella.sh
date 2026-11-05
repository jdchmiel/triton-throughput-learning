#!/bin/bash
# Stella-only Triton server. Starts a repository that exposes stella400m and
# nothing else, so the GPU is not shared with miniLM/sage/bert and the numbers
# in a sweep belong to this model alone.
#
#   ./run_triton_stella.sh              # foreground, Ctrl-C to stop
#   ./run_triton_stella.sh --bg         # detached, logs to /tmp/triton-stella.log
#   ./run_triton_stella.sh --stop       # stop it
#
# Single-model here means --load-model=stella400m, not a separate mount. The
# repository is still /opt/ml/model, so the model directory keeps sitting at
# /AI/models/triton/stella400m next to the others; explicit mode just loads one
# of them. --model-control-mode=explicit is what lets the sweep rewrite
# config.pbtxt and POST .../load between rows without restarting the server.
#
# --http-thread-count: the HTTP worker pool defaults to 8, and each in-flight
# request occupies one thread, so the default silently caps concurrency below
# what an instance_group.count sweep is trying to measure. Override with
# HTTP_THREADS when a run needs more in-flight requests than 16 -- the
# co-tenancy sweep offers 100 TPS of search on top of a heavy indexing load,
# which is well over 16 concurrent requests, and a starved worker pool would
# show up as queueing that belongs to the HTTP layer rather than the model.
HTTP_THREADS=${HTTP_THREADS:-16}
#
# --gpus all is required: the model is KIND_GPU and the python backend stub
# inherits the container's device visibility.

CONTAINER=triton-stella
IMAGE=763104351884.dkr.ecr.us-east-1.amazonaws.com/sagemaker-tritonserver:26.05-py3

if [ "$1" = "--stop" ]; then
  docker rm -f $CONTAINER 2>/dev/null || echo "no $CONTAINER container running"
  exit 0
fi

COMMON_ARGS="tritonserver --model-repository=/opt/ml/model \
  --model-control-mode=explicit \
  --load-model=stella400m \
  --strict-model-config=false \
  --http-thread-count=$HTTP_THREADS \
  --rate-limit=execution_count --log-verbose=1"

# Note: --rm is deliberately NOT in the base. A crash during model load (for
# example --http-thread-count=256, which segfaults Triton at exit 139) would
# otherwise delete the container before anyone could read its logs, and the
# readiness loop would report "No such container" instead of the real cause.
# --stop and the pre-clean below do the removal instead.
DOCKER_BASE="docker run --gpus all \
  --name $CONTAINER \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -p 8080:8080 -p 8081:8081 -p 8082:8082 \
  -v /AI/models/triton:/opt/ml/model \
  -e SAGEMAKER_BIND_TO_PORT=8080 \
  -e SAGEMAKER_SAFE_PORT_RANGE=8081-8082"

if [ "$1" = "--bg" ]; then
  # Detached so a sweep can drive it over ssh. -d returns as soon as the
  # container starts, which is NOT the same as the model being ready; callers
  # must poll /v2/models/stella400m/ready (stella takes ~25s to load).
  docker rm -f $CONTAINER >/dev/null 2>&1
  $DOCKER_BASE -d $IMAGE $COMMON_ARGS
  echo "waiting for stella400m to become ready..."
  for i in $(seq 1 60); do
    if curl -s -o /dev/null http://localhost:8000/v2/models/stella400m/ready; then
      echo "READY after ${i}s"
      exit 0
    fi
    sleep 1
  done
  # Distinguish "still loading" from "crashed": a dead container will never
  # become ready, and its exit code is the actual diagnosis.
  echo "TIMEOUT waiting for ready"
  docker inspect $CONTAINER --format "state={{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}}" 2>/dev/null
  echo "--- last log lines ---"
  docker logs --tail 30 $CONTAINER 2>&1
  exit 1
fi

# shellcheck disable=SC2086
$DOCKER_BASE -it $IMAGE $COMMON_ARGS
