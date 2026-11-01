#!/usr/bin/env python3
"""Closed-loop load generator for intent-replica.

Runs N concurrent workers that each send one utterance per request and
report throughput plus latency percentiles. Run it inside the Triton
container so the client NIC is not part of what is being measured -- the
box's other NIC is 100 Mbps full duplex, which caps request rate well below
what the model can serve.

    docker cp loadgen.py triton:/tmp/
    docker exec triton python3 /tmp/loadgen.py 16 20     # conc=16, 20s

Payloads rotate through a small set so the encoder sees length variation
without the client-side cost dominating the sample.
"""

import json
import sys
import threading
import time
import urllib.request

URL = "http://localhost:8000/v2/models/intent-replica/infer"

TEXTS = [
    "how do I reset my pin",
    "why was my card declined",
    "what is my balance",
    "I want to top up my account",
    "cancel my subscription please",
]

lock = threading.Lock()
latencies = []
stop_at = 0.0


def worker():
    i = 0
    while time.perf_counter() < stop_at:
        text = TEXTS[i % len(TEXTS)]
        i += 1
        body = json.dumps({
            "inputs": [{"name": "TEXT", "datatype": "BYTES",
                        "shape": [1, 1], "data": [text]}]
        }).encode()
        req = urllib.request.Request(
            URL, data=body, headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        urllib.request.urlopen(req).read()
        with lock:
            latencies.append(time.perf_counter() - t0)


def main():
    conc = int(sys.argv[1])
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0

    # Warmup pays the first-request CUDA context and autotune cost so the
    # measured window is steady state.
    global stop_at
    stop_at = time.perf_counter() + 3
    for _ in range(conc):
        worker()
    latencies.clear()

    stop_at = time.perf_counter() + duration
    threads = [threading.Thread(target=worker) for _ in range(conc)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    latencies.sort()
    n = len(latencies)
    print("RESULT conc=%d rps=%.0f p50=%.2fms p99=%.2fms n=%d" % (
        conc, n / wall,
        latencies[n // 2] * 1000,
        latencies[int(n * 0.99)] * 1000,
        n), flush=True)


if __name__ == "__main__":
    main()
