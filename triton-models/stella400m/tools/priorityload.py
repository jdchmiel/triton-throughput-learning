#!/usr/bin/env python3
"""Open-loop two-class load generator for the stella co-tenancy experiment.

Why this is not loadgen.py: loadgen.py is closed-loop -- you hand it a worker
count and it reports whatever rate those workers manage. That cannot answer a
co-tenancy question. Give indexing 200 workers and search 8 and indexing wins
by construction; the graph measures the harness, not the server. This generator
fixes the OFFERED rate of each class and measures what the server delivers,
which is the only way to see whether a 100 TPS search SLO survives a rising
indexing load.

It also carries the metric that caught the mistake in the first sweep:
`send_delay` is how long a request waited between its scheduled send time and
the moment a worker thread actually put it on the wire. If the client cannot
afford the offered rate, send_delay grows and every throughput number in the
run is a client ceiling pretending to be a server ceiling. A send_delay near
zero is what makes a "the server is the limit" claim safe to make.

Traffic classes, chosen to match the two real jobs:
  search    short query, USE_PROMPT=1, high priority -- the latency SLO
  indexing  long passage, USE_PROMPT=0, low priority -- the bulk work

Stdlib only: the container has no requests/aiohttp/tritonclient.

Emits one JSON object per second per class on stdout for the plotter, and a
single summary object on stderr for the sweep table.

Usage (inside the container):
  python3 priorityload.py --search-rate 100 --index-rate 0 --duration 30
"""

import argparse
import http.client
import json
import random
import sys
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

# Short queries -- the shape an intent classifier actually sees on the search
# path. Deliberately short: padding a search request out to 512 tokens would
# flatter the interference result.
QUERIES = [
    "how do I reset my password",
    "cancel my subscription",
    "where is my order",
    "update the billing email on my account",
    "why was my payment declined",
    "talk to a human",
    "change the shipping address",
    "request a refund for a damaged item",
    "is there a student discount",
    "how long does delivery take",
    "close my account",
    "download an invoice",
    "the app keeps crashing on startup",
    "transfer my balance to another account",
    "what are your business hours",
]

_WORDS = (
    "the customer agreement describes how usage based billing is calculated "
    "across tiers and what happens when a plan is changed partway through a "
    "billing cycle including prorated credits and the conditions under which "
    "those credits expire rather than rolling forward into the following "
    "period when an account has been flagged for review by the payments team "
    "and support agents are instructed to reference the applicable schedule "
    "before quoting any figure to the customer involved"
).split()


def make_passage(rng, words):
    """A long passage, so indexing really is the expensive job it is in prod.

    Cost in this model scales with the padded sequence length of the whole
    batch, so short indexing payloads would understate the interference the
    SLO has to survive. These land near the 512 token MAX_LENGTH.
    """
    return " ".join(rng.choice(_WORDS) for _ in range(words))


def encode_request(text, use_prompt, priority=None):
    req = {
        "inputs": [
            {"name": "TEXT", "shape": [1, 1], "datatype": "BYTES",
             "data": [text]},
            {"name": "USE_PROMPT", "shape": [1, 1], "datatype": "BOOL",
             "data": [bool(use_prompt)]},
        ]
    }
    if priority is not None:
        # Priority travels in the request body's "parameters" map. There is NO
        # priority HTTP header: in http_server.cc the only call to
        # TRITONSERVER_InferenceRequestSetPriorityUInt64 lives inside
        # ParseJsonTritonParams, so a header named `priority` is parsed as a
        # no-op and every request silently lands on default_priority_level.
        # Sending a header instead is invisible -- the server accepts it and
        # both arms of a priority experiment behave identically.
        req["parameters"] = {"priority": int(priority)}
    return json.dumps(req).encode()


def post(host, port, model, body, timeout=10.0):
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    headers = {"Content-Type": "application/json"}
    try:
        conn.request("POST", "/v2/models/%s/infer" % model, body=body,
                     headers=headers)
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


class Recorder:
    """Thread-safe sample store. Each sample is (offered_t, lat, cap, pool, rate).

    The wait is split in two because the two halves mean opposite things.
    cap_wait is time a request spent waiting for one of the bounded in-flight
    slots -- that is the SERVER failing to keep up, and it is the signal we are
    looking for. pool_wait is time waiting for a worker thread -- that would be
    the CLIENT failing to keep up, which would make every throughput number a
    lie. Only a near-zero pool_wait licenses reading cap_wait as server
    saturation.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.samples = []
        self.errors = 0

    def add(self, offered_t, latency, cap_wait, pool_wait, rate=0.0):
        with self.lock:
            self.samples.append(
                (offered_t, latency, cap_wait, pool_wait, rate))

    def fail(self):
        with self.lock:
            self.errors += 1


def worker(host, port, model, body, rec, t0, offered, cap_wait,
           slot, rate=0.0):
    pool_wait = time.perf_counter() - t0 - offered - cap_wait
    send = time.perf_counter()
    try:
        try:
            status, payload = post(host, port, model, body)
            if status != 200 or not payload:
                rec.fail()
            else:
                rec.add(offered, time.perf_counter() - send, cap_wait,
                        pool_wait, rate)
        finally:
            slot.release()
    except Exception:
        rec.fail()


class Rate:
    """Offered rate the scheduler loop re-reads, so it can change mid-run."""

    def __init__(self, value, max_value=None):
        self.lock = threading.Lock()
        self.value = value
        # Pool sizing has to anticipate a ramp; growing a ThreadPoolExecutor
        # mid-run is not possible.
        self.max_value = max_value if max_value is not None else value

    def get(self):
        with self.lock:
            return self.value

    def set(self, value):
        with self.lock:
            self.value = value


def run_class(name, rate, duration, host, port, model, bodies, t0,
              rec, max_inflight=0):
    """Offered rate at absolute deadlines; never waits on a response.

    The rate is re-read every dispatch so a coordinator can step it during the
    run. Stepping inside one run is what makes the graph a continuous ramp
    instead of a row of disconnected bars.
    """
    rng = random.Random(abs(hash(name)) & 0xFFFF)
    next_due = 0.0
    # Bounded in-flight. Open-loop load above the server's capacity otherwise
    # queues work without limit: memory grows, and the executor drains every
    # queued request at shutdown, turning a short step into a multi-minute
    # tail. The cap also produces the measurement we want -- time spent
    # waiting for a slot is the server not keeping up.
    cap = max_inflight or max(64, int(rate.max_value * 2))
    slot = threading.Semaphore(cap)
    with ThreadPoolExecutor(max_workers=int(rate.max_value * 4) + 32) as pool:
        while True:
            now = time.perf_counter() - t0
            if now >= duration:
                break
            current = rate.get()
            # A ramp can step this class to zero. Idle rather than divide by
            # zero, and keep the deadline pinned so the step back up is
            # immediate instead of paying for the idle time in the schedule.
            if current <= 0:
                next_due = now
                time.sleep(0.05)
                continue
            if next_due > now:
                time.sleep(min(next_due - now, 0.05))
                continue
            offered = next_due
            # Take a slot before submitting, so the executor queue stays empty
            # and the wait is attributable to the request that paid for it.
            while not slot.acquire(timeout=0.05):
                if time.perf_counter() - t0 >= duration + 30:
                    break
            cap_wait = time.perf_counter() - t0 - offered
            body = bodies[rng.randrange(len(bodies))]
            # `current` is the OFFERED rate at the moment of dispatch, carried
            # through to the bucket. The graph needs offered rather than
            # delivered: delivered flattens out once the server saturates, so
            # plotting it would show the indexing load stopping its climb
            # exactly when the interesting part of the experiment begins.
            pool.submit(worker, host, port, model, body, rec,
                        t0, offered, cap_wait, slot, current)
            next_due += 1.0 / current
        # In-flight requests are allowed to land -- their latency belongs in
        # the numbers -- so the executor drains rather than drops. The cap
        # keeps that drain bounded.


def pct(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(int(len(s) * p / 100.0), len(s) - 1)] * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="stella400m")
    ap.add_argument("--search-rate", type=float, required=True)
    ap.add_argument("--index-rate", type=float, default=0.0,
                    help="flat offered indexing rate; ignored when --index-ramp is given")
    # Triton priority levels start at 1 and 1 is the HIGHEST priority
    # (model_config.proto), so search gets 1 and indexing gets 2.
    ap.add_argument("--search-priority", type=int, default=1)
    ap.add_argument("--index-priority", type=int, default=2)
    ap.add_argument("--no-priority", action="store_true",
                    help="send no priority on either class; the "
                         "control that shows whether priority is doing any "
                         "work at all")
    ap.add_argument("--duration", type=float, default=30)
    ap.add_argument("--warmup", type=float, default=3)
    ap.add_argument("--passage-words", type=int, default=380)
    for which, dflt in (("search", 0), ("indexing", 0)):
        ap.add_argument(f"--{which}-max-inflight", type=int, default=dflt,
                        help=f"cap on concurrent in-flight {which} requests "
                             "(0 = 2x the peak offered rate). Keep the sum "
                             "across both classes below the server's "
                             "--http-thread-count: queueing beyond that "
                             "happens in the socket backlog, which shows up "
                             "as latency and gets misattributed to the model.")
    ap.add_argument("--bucket-seconds", type=float, default=5.0,
                    help="width of each reported time bucket. Keep this at or "
                         "above the ramp step so every bucket sits entirely "
                         "inside one load step; a bucket that straddles a step "
                         "change blends two loads into one percentile.")
    ap.add_argument("--label", default="")
    ap.add_argument("--index-ramp", default="",
                    help="comma-separated offered indexing rates, one per "
                         "RAMP-STEP seconds, e.g. '0,20,40,60'. Overrides "
                         "--index-rate for the schedule; --index-rate still "
                         "sizes the thread pool.")
    ap.add_argument("--ramp-step", type=float, default=10)
    args = ap.parse_args()

    u = urllib.parse.urlparse(args.url)
    host, port = u.hostname, u.port or 80

    ramp = [float(x) for x in args.index_ramp.split(",") if x.strip()] \
        if args.index_ramp else []

    rng = random.Random(7)
    # Priority is a property of the body, because that is where the server
    # reads it from -- so it is baked in here rather than threaded down to the
    # socket layer, where it would be tempting to send as a header.
    search_prio = None if args.no_priority else args.search_priority
    index_prio = None if args.no_priority else args.index_priority
    search_bodies = [encode_request(q, True, search_prio) for q in QUERIES]
    index_bodies = [
        encode_request(make_passage(rng, args.passage_words + d), False,
                       index_prio)
        for d in (0, 40, 80, -40)
    ]

    if args.warmup > 0:
        t_end = time.perf_counter() + args.warmup
        while time.perf_counter() < t_end:
            try:
                post(host, port, args.model, search_bodies[0])
            except Exception:
                pass
            time.sleep(0.05)

    search_rate = Rate(args.search_rate)
    # Size the pool for the loudest step, not the starting rate, or a ramp
    # would be silently throttled by the pool and look like a server ceiling.
    peak = max(ramp) if ramp else args.index_rate
    index_rate = Rate(ramp[0] if ramp else args.index_rate, peak)

    recs = {}
    threads = []
    t0 = time.perf_counter()
    caps = {"search": args.search_max_inflight,
            "indexing": args.indexing_max_inflight}
    for name, rate_obj, bodies in (
        ("search", search_rate, search_bodies),
        ("indexing", index_rate, index_bodies),
    ):
        if rate_obj.get() <= 0 and not (name == "indexing" and ramp):
            continue
        recs[name] = Recorder()
        th = threading.Thread(
            target=run_class,
            args=(name, rate_obj, args.duration, host, port, args.model,
                  bodies, t0, recs[name], caps[name]),
        )
        th.start()
        threads.append(th)

    if ramp and "indexing" in recs:
        def coordinator():
            for i, value in enumerate(ramp):
                target = i * args.ramp_step
                while True:
                    now = time.perf_counter() - t0
                    if now >= args.duration:
                        return
                    if now >= target:
                        break
                    time.sleep(min(target - now, 0.05))
                # Set 0 literally, not a small floor: run_class parks the
                # deadline at `now` while idle, so the step back up fires on
                # the next tick. A floor would instead schedule the next
                # dispatch thousands of seconds out and strand the class.
                index_rate.set(value)
        cth = threading.Thread(target=coordinator)
        cth.start()
        threads.append(cth)

    for th in threads:
        th.join()

    # Attribute each completion to the bucket in which it was offered: that is
    # what "indexing requests in flight during second N" means on the graph.
    #
    # Bucket width matters more than it looks. A 1s bucket holds ~100 search
    # samples, and the p99 of 100 samples is the 99th value -- it jumps around
    # by tens of milliseconds purely from sampling noise. Plotting that as the
    # headline makes a stable service look unstable. Wider buckets trade time
    # resolution for a percentile that is actually comparable between steps,
    # so the default is 5s and the plotter is told which width produced it.
    bw = args.bucket_seconds
    buckets = {}
    for name, rec in recs.items():
        for offered, lat, cap_wait, pool_wait, rate in rec.samples:
            sec = int(offered / bw) * bw
            b = buckets.setdefault(sec, {}).setdefault(
                name, {"lat": [], "cap": [], "pool": [], "off": [],
                       "offered": 0})
            b["lat"].append(lat)
            b["cap"].append(cap_wait)
            b["pool"].append(pool_wait)
            b["offered"] = rate

    for sec in sorted(buckets):
        for name in sorted(buckets[sec]):
            b = buckets[sec][name]
            # The final bucket is usually partial: the run stops offering at
            # --duration, which need not land on a bucket boundary. Dividing a
            # partial bucket's count by the full width understates the rate --
            # the last indexing bucket here read 31.9 rps for a steady ~80 rps
            # stream, which looks like a collapse on the graph and is only the
            # window being short.
            eff = min(bw, max(args.duration - sec, 1e-9))
            print(json.dumps({
                "label": args.label,
                "t": sec,
                "class": name,
                # Delivered RATE, not a count. With 1s buckets the two are the
                # same number, which is why this is easy to miss, but a 15s
                # bucket would plot search at 1500 TPS.
                "tps": round(len(b["lat"]) / eff, 1),
                "bucket_seconds": round(eff, 3),
                "offered_rate": round(b["offered"], 1),
                "samples": len(b["lat"]),
                "p50": round(pct(b["lat"], 50), 2),
                "p95": round(pct(b["lat"], 95), 2),
                "p99": round(pct(b["lat"], 99), 2),
                "max": round(max(b["lat"]) * 1000, 2),
                "cap_wait_max": round(max(b["cap"]) * 1000, 2),
                "pool_wait_max": round(max(b["pool"]) * 1000, 2),
            }))

    summary = {"label": args.label, "class": "summary",
               "bucket_seconds": bw,
               "duration": args.duration,
               "search_offered": args.search_rate,
               "indexing_offered": (args.index_ramp or args.index_rate),
               "priority": "off" if args.no_priority else "on"}
    for name, key in (("search", "search"), ("indexing", "indexing")):
        rec = recs.get(name)
        lat = [s[1] for s in rec.samples] if rec else []
        capw = [s[2] for s in rec.samples] if rec else []
        poolw = [s[3] for s in rec.samples] if rec else []
        summary["%s_delivered" % key] = round(len(lat) / args.duration, 1)
        summary["%s_p50" % key] = round(pct(lat, 50), 2)
        summary["%s_p99" % key] = round(pct(lat, 99), 2)
        summary["%s_errors" % key] = rec.errors if rec else 0
        # Server-side saturation: requests that had to wait for an in-flight
        # slot before they could even be sent.
        summary["%s_cap_wait_max" % key] = (
            round(max(capw) * 1000, 2) if capw else 0)
        # The client-ceiling guard: if this is not ~0, the numbers are the
        # client's limit, not the server's.
        summary["%s_pool_wait_max" % key] = (
            round(max(poolw) * 1000, 2) if poolw else 0)
    print(json.dumps(summary), file=sys.stderr)


if __name__ == "__main__":
    main()
