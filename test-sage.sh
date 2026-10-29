curl -X POST http://192.168.1.151:8000/v2/models/miniLM/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "TEXT",
        "shape":[2,1],
        "datatype": "BYTES",
        "data": [
		"This is a test sentence to generate embeddings.",
		"This is another sentence to test with"
	]
      }
    ]
  }'
