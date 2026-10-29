# this skips sagemaker entrypoint entierely to try and get multi model working
# https://aws.github.io/deep-learning-containers/reference/available_images/
#  single model mode specify one like:  -e SAGEMAKER_TRITON_DEFAULT_MODEL_NAME=miniLM \
#
# The three flags below are what make `campaign.py` possible:
#   --model-control-mode=explicit   allows runtime POST .../load and .../unload.
#                                   Without it every load call fails with
#                                   "explicit model load / unload is not allowed
#                                   if polling is enabled".
#   --load-model='*'                explicit mode loads NOTHING by default, so
#                                   this restores the old "start everything"
#                                   behaviour at boot. Keep the quotes: the
#                                   value must reach Triton as a literal '*',
#                                   and it must be the only --load-model flag.
#   --strict-model-config=false     accept configs pushed through the load API
#                                   without demanding every field be on disk.
# -p 8002:8002 publishes the Prometheus metrics endpoint, which is where
# nv_inference_pending_request_count (queue depth) lives.

docker run --gpus all --rm -it \
  -p 8000:8000 \
  -p 8002:8002 \
  -p 8080:8080 \
  -p 8081:8081 \
  -p 8082:8082 \
  -v /AI/models/triton:/opt/ml/model \
  -e SAGEMAKER_BIND_TO_PORT=8080 \
  -e SAGEMAKER_SAFE_PORT_RANGE=8081-8082 \
  -e SAGEMAKER_MULTI_MODEL=true \
  763104351884.dkr.ecr.us-east-1.amazonaws.com/sagemaker-tritonserver:26.05-py3 \
  tritonserver --model-repository=/opt/ml/model \
    --model-control-mode=explicit \
    --load-model='*' \
    --strict-model-config=false \
    --rate-limit=execution_count --log-verbose=1
