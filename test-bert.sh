#!/bin/bash
# bert-base-uncased on Triton (onnxruntime backend).
#
# The ONNX graph is the raw BERT MLM head, so it returns per-token vocab
# logits with shape [batch, seq_len, 30522] -- NOT pooled embeddings.
# That output is ~600 MB per request at seq_len 128, so keep seq_len small
# unless you add a pooling/argmax node to the model.
#
# Inputs are token ids, so run them through a tokenizer first. For a
# text-in variant that tokenizes server-side, see test-intent.sh.

curl -X POST http://192.168.1.151:8000/v2/models/bert-base-uncased/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      { "name": "input_ids", "shape": [ 1, 5 ], "datatype": "INT64", "data": [ 101, 2023, 2003, 1037, 102 ] },
      { "name": "attention_mask", "shape": [ 1, 5 ], "datatype": "INT64", "data": [ 1, 1, 1, 1, 1 ] },
      { "name": "token_type_ids", "shape": [ 1, 5 ], "datatype": "INT64", "data": [ 0, 0, 0, 0, 0 ] }
    ]
  }'
