curl -X POST http://192.168.1.151:8000/v2/models/miniLM-onnx/infer \
  -H "Content-Type: application/json" \
  -H "X-Amzn-SageMaker-Target-Model: miniLM-onnx" \
  -d '{
    "inputs": [
      { "name": "input_ids", "shape": [ 1, 5 ], "datatype": "INT64", "data": [ 101, 2023, 2003, 1037, 102 ] },
      { "name": "attention_mask", "shape": [ 1, 5 ], "datatype": "INT64", "data": [ 1, 1, 1, 1, 1 ] },
      { "name": "token_type_ids", "shape": [ 1, 5 ], "datatype": "INT64", "data": [ 0, 0, 0, 0, 0 ] }
    ]
  }'
