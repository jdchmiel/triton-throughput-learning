import asyncio
import time
import aiohttp

# --- SERVER URL SETTINGS ---
# Native Triton V2 URL endpoints
URL_PYTHON = "http://localhost:8000/v2/models/miniLM/infer"
URL_ONNX = "http://localhost:8000/v2/models/miniLM-onnx/infer"

# --- BYPASS SYSTEM FILTER: EXPLICIT INPUT PARAMETERS ---
# Array values are assigned clearly to variables to avoid interface redaction bugs
BATCH_SIZE = 1
TOKEN_COUNT = 5

# Safe numeric arrays for shape properties
SHAPE_FOR_STRING_MODEL = [ BATCH_SIZE, BATCH_SIZE ]  # Evaluates to 1 by 1 matrix
SHAPE_FOR_ONNX_MODEL = [ BATCH_SIZE, TOKEN_COUNT ]   # Evaluates to 1 by 5 matrix

# Numeric payload inputs
INPUT_IDS_DATA = [ 101, 2023, 2003, 1037, 102 ]
ATTENTION_MASK_DATA = [ 1, 1, 1, 1, 1 ]
TOKEN_TYPE_IDS_DATA = [ 0, 0, 0, 0, 0 ]

# --- PAYLOAD GENERATION ---
PAYLOAD_PYTHON = {
    "inputs": [
        {
            "name": "TEXT",
            "shape": SHAPE_FOR_STRING_MODEL,
            "datatype": "BYTES",
            "data": ["This is a test sentence to generate embeddings."]
        }
    ]
}

PAYLOAD_ONNX = {
    "inputs": [
        { "name": "input_ids", "shape": SHAPE_FOR_ONNX_MODEL, "datatype": "INT64", "data": INPUT_IDS_DATA },
        { "name": "attention_mask", "shape": SHAPE_FOR_ONNX_MODEL, "datatype": "INT64", "data": ATTENTION_MASK_DATA },
        { "name": "token_type_ids", "shape": SHAPE_FOR_ONNX_MODEL, "datatype": "INT64", "data": TOKEN_TYPE_IDS_DATA }
    ]
}

async def send_request(session, url, json_data, req_id):
    """Sends a single async inference POST request to Triton and measures time."""
    start_time = time.perf_counter()
    try:
        async with session.post(url, json=json_data) as response:
            status = response.status
            # Read response text to ensure data transfer completes fully
            await response.text()
            end_time = time.perf_counter()
            elapsed = (end_time - start_time) * 1000.0
            return url.split('/')[-2], req_id, status, elapsed
    except Exception as e:
        end_time = time.perf_counter()
        return url.split('/')[-2], req_id, "ERROR", (end_time - start_time) * 1000.0

async def main():
    # Number of concurrent workers hitting each model endpoint simultaneously
    CONCURRENT_REQUESTS_PER_MODEL = 200
    
    print(f"Began dispatching {CONCURRENT_REQUESTS_PER_MODEL * 2} concurrent requests to Triton...")
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        # Stagger a flood of mixed requests into the active task group loop
        for idx in range(CONCURRENT_REQUESTS_PER_MODEL):
            tasks.append(send_request(session, URL_ONNX, PAYLOAD_ONNX, idx))
            tasks.append(send_request(session, URL_PYTHON, PAYLOAD_PYTHON, idx))
            
        # Execute all tasks in parallel across your threads
        results = await asyncio.gather(*tasks)
        
    # --- PROCESS AND DISPLAY PERFORMANCE TIMINGS ---
    onnx_times = [r[3] for r in results if r[0] == "miniLM-onnx" and r[2] == 200]
    python_times = [r[3] for r in results if r[0] == "miniLM" and r[2] == 200]
    
    print("\n================ TESTING SUMMARY ================")
    if onnx_times:
        print(f"ONNX Model (Priority 1) -> Total Success: {len(onnx_times)}")
        print(f"   Avg Latency: {sum(onnx_times)/len(onnx_times):.2f} ms")
        print(f"   Min Latency: {min(onnx_times):.2f} ms | Max Latency: {max(onnx_times):.2f} ms")
    else:
        print("ONNX Model Execution Failed completely.")
        
    print("-------------------------------------------------")
    if python_times:
        print(f"Python Model (Priority 2) -> Total Success: {len(python_times)}")
        print(f"   Avg Latency: {sum(python_times)/len(python_times):.2f} ms")
        print(f"   Min Latency: {min(python_times):.2f} ms | Max Latency: {max(python_times):.2f} ms")
    else:
        print("Python Model Execution Failed completely.")
    print("=================================================")

if __name__ == "__main__":
    asyncio.run(main())
