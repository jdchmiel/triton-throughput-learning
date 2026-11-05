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
# --http-thread-count=16: the HTTP worker pool defaults to 8, and each
# in-flight request occupies one thread, so the default silently caps
# concurrency below what an instance_group.count sweep is trying to measure.
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
  --http-thread-count=16 \
  --rate-limit=execution_count --log-verbose=1"

DOCKER_BASE="docker run --gpus all --rm \
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
  $DOCKER_BASE -d $IMAGE $COMMON_ARGS
  echo "waiting for stella400m to become ready..."
  for i in $(seq 1 60); do
    if curl -s -o /dev/null http://localhost:8000/v2/models/stella400m/ready; then
      echo "READY after ${i}s"
      exit 0
    fi
    sleep 1
  done
  echo "TIMEOUT waiting for ready; last log lines:"
  docker logs --tail 30 $CONTAINER
  exit 1
fi

# shellcheck disable=SC2086
$DOCKER_BASE -it $IMAGE $COMMON_ARGS
