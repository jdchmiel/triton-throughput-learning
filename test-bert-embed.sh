#!/bin/bash
# bert-base-uncased-embed on Triton (onnxruntime backend).
#
# Trimmed graph: MLM vocab head removed, attention-masked mean pooling added
# in-graph. Returns [batch, 768] embeddings.
#
# Inputs are token ids, so tokenize client-side. For a TEXT-in variant via
# the python backend, see test-intent.sh.

curl -s -X POST http://192.168.1.151:8000/v2/models/bert-base-uncased-embed/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      { "name": "input_ids", "shape": [ 1, 5 ], "datatype": "INT64", "data": [ 101, 2023, 2003, 1037, 102 ] },
      { "name": "attention_mask", "shape": [ 1, 5 ], "datatype": "INT64", "data": [ 1, 1, 1, 1, 1 ] },
      { "name": "token_type_ids", "shape": [ 1, 5 ], "datatype": "INT64", "data": [ 0, 0, 0, 0, 0 ] }
    ]
  }' | python3 -c '
import sys, json
d = json.load(sys.stdin)
for o in d["outputs"]:
    v = o["data"]
    print(o["name"], o["datatype"], o["shape"])
    print("  first 8:", [round(x, 4) for x in v[:8]])
    print("  L2 norm:", round(sum(x * x for x in v) ** 0.5, 4))
'
