#!/usr/bin/env python3
"""Behavioural check that Triton priority scheduling is actually in effect.

Why this exists instead of a config assertion
---------------------------------------------
The obvious guard -- send a priority outside [1, priority_levels] and require a
rejection -- does not work. Triton clamps an out-of-range level instead of
refusing the request, so `parameters.priority: 99` returns HTTP 200 even with
`priority_levels: 2` live. A guard written that way aborts a *healthy* server.

The only thing worth knowing is whether a high-priority request overtakes
low-priority work that is already queued. That is observable, so measure it:
saturate the model with low-priority traffic, then time a high-priority probe
against a control probe that carries no priority at all.

Why the queue has to be deep
----------------------------
The number of low-priority senders is not cosmetic. If the backlog is smaller
than one batch, the batcher puts the probe in the SAME batch as the
low-priority work, and reordering the queue cannot help -- the probe is already
running. Measured on this box with max_batch_size 128 and 40 senders: 609 ms
with priority, 596 ms without, i.e. "priority does nothing". With
max_batch_size 16 and 100 senders: 392 ms with priority, 1741 ms without. Same
server, same request body, opposite conclusions.

So this probe deliberately over-offers, and it reports the queue depth it
achieved so a future reader can tell "priority is off" from "there was nothing
to reorder".

Exit code 0 = priority is live, 1 = it is not, 2 = could not measure.
"""

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

PASSAGE = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
    "omicron pi rho sigma tau upsilon phi chi psi omega " * 32
)


def encode(text, priority):
    req = {
        "inputs": [
            {"name": "TEXT", "shape": [1, 1], "datatype": "BYTES", "data": [text]},
            {
                "name": "USE_PROMPT",
                "shape": [1, 1],
                "datatype": "BOOL",
                "data": [False],
            },
        ]
    }
    # Priority travels in the request body. There is no priority HTTP header --
    # http_server.cc reads it only from "parameters", and a header is accepted
    # and silently ignored, which is how an entire sweep once came out showing
    # two identical arms.
    if priority is not None:
        req["parameters"] = {"priority": int(priority)}
    return json.dumps(req).encode()


def post(url, body, timeout):
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp.read()
    return (time.perf_counter() - t0) * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="stella400m")
    ap.add_argument("--senders", type=int, default=100,
                    help="low-priority senders; must exceed one batch or the "
                         "probe cannot demonstrate reordering")
    ap.add_argument("--high-priority", type=int, default=1)
    ap.add_argument("--low-priority", type=int, default=2)
    ap.add_argument("--settle", type=float, default=5.0,
                    help="seconds of saturation before probing")
    ap.add_argument("--probes", type=int, default=8)
    ap.add_argument("--gap", type=float, default=0.4)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--require-ratio", type=float, default=0.75,
                    help="high-priority p50 must be below this fraction of "
                         "the control p50 for the check to pass")
    args = ap.parse_args()

    url = f"{args.url}/v2/models/{args.model}/infer"
    low_body = encode(PASSAGE, args.low_priority)
    high_body = encode("how do I reset my pin", args.high_priority)
    # The control carries no priority, so it lands on default_priority_level.
    # In the sweep that is deliberately the LOW level, which makes this the
    # honest comparison: what a caller who knows nothing about priorities gets.
    ctrl_body = encode("how do I reset my pin", None)

    stop = threading.Event()
    errors = []

    def hog():
        while not stop.is_set():
            try:
                post(url, low_body, args.timeout)
            except Exception as exc:  # noqa: BLE001 - counted, not raised
                errors.append(str(exc))
                time.sleep(0.05)

    threads = [threading.Thread(target=hog, daemon=True) for _ in range(args.senders)]
    for t in threads:
        t.start()
    stop.wait(args.settle)

    def sample(body):
        out = []
        for _ in range(args.probes):
            try:
                out.append(post(url, body, args.timeout))
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
            time.sleep(args.gap)
        return sorted(out)

    high = sample(high_body)
    ctrl = sample(ctrl_body)
    stop.set()
    for t in threads:
        t.join(timeout=args.timeout)

    if not high or not ctrl:
        print(json.dumps({"ok": False, "error": "no samples",
                          "errors": errors[:5]}))
        return 2

    hp = statistics.median(high)
    cp = statistics.median(ctrl)
    ratio = hp / cp if cp else float("inf")
    ok = ratio <= args.require_ratio
    print(json.dumps({
        "ok": ok,
        "high_p50_ms": round(hp, 1),
        "control_p50_ms": round(cp, 1),
        "ratio": round(ratio, 3),
        "high_max_ms": round(high[-1], 1),
        "control_max_ms": round(ctrl[-1], 1),
        "senders": args.senders,
        "errors": len(errors),
    }))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
