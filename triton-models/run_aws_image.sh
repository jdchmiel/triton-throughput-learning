# https://aws.github.io/deep-learning-containers/reference/available_images/
#  single model mode specify one like:  -e SAGEMAKER_TRITON_DEFAULT_MODEL_NAME=miniLM \

docker run --gpus all --rm -it \
  -p 8080:8080 \
  -p 8081:8081 \
  -p 8082:8082 \
  -v /AI/models/triton:/opt/ml/model \
  -e SAGEMAKER_BIND_TO_PORT=8080 \
  -e SAGEMAKER_SAFE_PORT_RANGE=8081-8082 \
  -e SAGEMAKER_MULTI_MODEL=true \
  763104351884.dkr.ecr.us-east-1.amazonaws.com/sagemaker-tritonserver:26.05-py3 \
  serve

