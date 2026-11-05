#!/usr/bin/env python3
"""Closed-loop load generator for stella400m.

Runs N concurrent workers that each send one string per request and report
throughput plus latency percentiles. Run it inside the Triton container so the
client NIC is not part of what is being measured -- the box's other NIC is
100 Mbps full duplex, which caps request rate well below what the model can
serve.

    docker cp loadgen.py triton:/tmp/
    docker exec triton python3 /tmp/loadgen.py 16 20     # conc=16, 20s

Payloads rotate through a set with real length variation. Stella truncates at
MAX_LENGTH and its cost scales with padded sequence length, so a uniform
payload set would understate what batching buys: the win comes from grouping
short requests with short requests.
"""

import json
import sys
import threading
import time
import urllib.request

URL = "http://localhost:8000/v2/models/stella400m/infer"

TEXTS = [
    "how do I reset my pin",
    "why was my card declined",
    "what is my balance",
    "I want to top up my account",
    "cancel my subscription please",
    "I recently made a transaction to a merchant I do not recognize and I "
    "would like to dispute the charge because I never authorized it and the "
    "merchant is not responding to my emails asking for a refund",
    "what is the exchange fee for converting euros",
    "my card expired last week and I still have not received the replacement",
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
    print("RESULT conc=%d rps=%.1f p50=%.2fms p99=%.2fms n=%d" % (
        conc, n / wall,
        latencies[n // 2] * 1000,
        latencies[int(n * 0.99)] * 1000,
        n), flush=True)


if __name__ == "__main__":
    main()
