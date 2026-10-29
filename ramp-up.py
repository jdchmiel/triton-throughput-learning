"""
ramp-up.py — Triton staged concurrency (ramp-up) throughput tester.

Standalone benchmark that keeps a pool of open, keep-alive connections to each
Triton endpoint and walks the pool size upward in fixed steps:

    1, 6, 11, ... 96, 100        (START_WORKERS + n * WORKER_INCREMENT, capped
                                  by a shortened final step onto MAX_WORKERS)

Each stage hammers every endpoint for STAGE_DURATION_SECONDS, then reports
throughput / latency / failure metrics. The ramp stops early when throughput
stops improving (plateau) or when too many requests fail.

Results are printed as a table, written to a CSV, and rendered to PNG line
graphs with Matplotlib. No web app, no server, just images on disk.

Every constant below is also a command-line flag, so the script stays usable
standalone while `campaign.py` can drive it non-interactively:

    python ramp-up.py --max 61 --duration 10 --output-dir experiments/.../exp-01_900rps

Alongside the client-side timings this also samples Triton's own per-model
statistics (`GET /v2/models/{name}/stats`) around each stage and records the
server-side view: realised average batch size, and the queue-vs-compute split.
That is the evidence the dynamic-batching experiments are actually about.

Dependencies: aiohttp, matplotlib
"""

import argparse
import asyncio
import csv
import io
import json
import math
import statistics
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiohttp

import matplotlib

matplotlib.use("Agg")  # headless: render straight to PNG files
import matplotlib.pyplot as plt  # noqa: E402

# =============================================================================
# == ADJUSTABLE CONSTANTS ====================================================
# =============================================================================

# --- SERVER / ROUTING --------------------------------------------------------
# SageMaker-hosted Triton (see test-onnx.sh). Change TRITON_BASE_URL to point
# anywhere else, e.g. "http://localhost:8000" for a plain Triton server.
TRITON_BASE_URL = "http://192.168.1.151:8000"

# SageMaker serving needs the target-model header routed per request. Plain
# Triton ignores it, so it is harmless either way; set to {} to drop it.
SAGEMAKER_TARGET_HEADER = "X-Amzn-SageMaker-Target-Model"

# --- RAMP SHAPE --------------------------------------------------------------
START_WORKERS = 1               # pool size (concurrent connections) per endpoint
WORKER_INCREMENT = 5            # connections added to the pool at each stage
MAX_WORKERS = 100               # stop ramping once the pool reaches this size
STAGE_DURATION_SECONDS = 10.0   # measured load applied at every stage
WARMUP_SECONDS = 1.0            # load applied but NOT counted, per stage

# --- EARLY-STOP RULES --------------------------------------------------------
PLATEAU_THRESHOLD = 0.02        # <2% aggregate throughput gain == "no gain"
PLATEAU_STAGES_REQUIRED = 2     # that many consecutive flat stages ends the ramp
FAILURE_STOP_RATE = 0.10        # >=10% failed requests in a stage ends the ramp
MIN_REQUESTS_PER_STAGE = 20     # ignore plateau logic on stages this small

# Concurrency levels always included in the ramp, and reported separately.
# Peak RPS alone is a weak ratchet: a config that peaks at 100 connections is
# operationally worse than one that peaks at 20, so these fixed reference
# points give campaign.py a tiebreaker that does not depend on where the peak
# happened to land.
REFERENCE_CONCURRENCIES = (8, 16, 32)

# --- CLIENT BEHAVIOUR --------------------------------------------------------
REQUEST_TIMEOUT_SECONDS = 30.0
KEEPALIVE_TIMEOUT_SECONDS = 75.0
POLL_INTERVAL_SECONDS = 0.01    # how often an idle worker checks for a new stage

# --- SERVER-SIDE STATS -------------------------------------------------------
# Triton updates its per-model stats AFTER the response reaches the client, so
# an "after" snapshot taken immediately under-counts the tail of the stage.
# Triton's own QA retries for up to 10s; this is the settle window we allow.
STATS_SETTLE_SECONDS = 2.0

# --- OUTPUT ------------------------------------------------------------------
OUTPUT_DIR = Path("ramp-results")
CSV_NAME = "ramp-results.csv"
PLOT_DPI = 130

# --- PAYLOADS (mirrored from benchy.py) --------------------------------------
BATCH_SIZE = 1
TOKEN_COUNT = 5

SHAPE_FOR_STRING_MODEL = [BATCH_SIZE, BATCH_SIZE]
SHAPE_FOR_ONNX_MODEL = [BATCH_SIZE, TOKEN_COUNT]

INPUT_IDS_DATA = [101, 2023, 2003, 1037, 102]
ATTENTION_MASK_DATA = [1, 1, 1, 1, 1]
TOKEN_TYPE_IDS_DATA = [0, 0, 0, 0, 0]

PAYLOAD_PYTHON = {
    "inputs": [
        {
            "name": "TEXT",
            "shape": SHAPE_FOR_STRING_MODEL,
            "datatype": "BYTES",
            "data": ["This is a test sentence to generate embeddings."],
        }
    ]
}

PAYLOAD_ONNX = {
    "inputs": [
        {"name": "input_ids", "shape": SHAPE_FOR_ONNX_MODEL, "datatype": "INT64", "data": INPUT_IDS_DATA},
        {"name": "attention_mask", "shape": SHAPE_FOR_ONNX_MODEL, "datatype": "INT64", "data": ATTENTION_MASK_DATA},
        {"name": "token_type_ids", "shape": SHAPE_FOR_ONNX_MODEL, "datatype": "INT64", "data": TOKEN_TYPE_IDS_DATA},
    ]
}


@dataclass(frozen=True)
class Endpoint:
    """One Triton inference endpoint plus the payload used to drive it."""

    key: str
    label: str
    model: str
    url: str
    payload: dict
    headers: dict = field(default_factory=dict)


def build_endpoints(base_url: str, target_header: str) -> list:
    """Build the endpoint list for a given Triton base URL."""
    return [
        Endpoint(
            key="onnx",
            label="miniLM-onnx (Priority 1)",
            model="miniLM-onnx",
            url=f"{base_url}/v2/models/miniLM-onnx/infer",
            payload=PAYLOAD_ONNX,
            headers={target_header: "miniLM-onnx"} if target_header else {},
        ),
        Endpoint(
            key="python",
            label="miniLM python (Priority 2)",
            model="miniLM",
            url=f"{base_url}/v2/models/miniLM/infer",
            payload=PAYLOAD_PYTHON,
        ),
    ]


ENDPOINTS = build_endpoints(TRITON_BASE_URL, SAGEMAKER_TARGET_HEADER)


@dataclass
class RunConfig:
    """Every tunable knob for one benchmark run.

    Defaults are the module constants above; ``parse_args`` overrides them from
    the command line so ``campaign.py`` can drive this script without editing it.
    """

    base_url: str = TRITON_BASE_URL
    target_header: str = SAGEMAKER_TARGET_HEADER
    start_workers: int = START_WORKERS
    worker_increment: int = WORKER_INCREMENT
    max_workers: int = MAX_WORKERS
    stage_duration: float = STAGE_DURATION_SECONDS
    warmup: float = WARMUP_SECONDS
    plateau_threshold: float = PLATEAU_THRESHOLD
    plateau_stages: int = PLATEAU_STAGES_REQUIRED
    failure_stop_rate: float = FAILURE_STOP_RATE
    min_requests_per_stage: int = MIN_REQUESTS_PER_STAGE
    request_timeout: float = REQUEST_TIMEOUT_SECONDS
    keepalive_timeout: float = KEEPALIVE_TIMEOUT_SECONDS
    output_dir: Path = OUTPUT_DIR
    label: str = ""
    only: str = ""
    collect_stats: bool = True
    stats_settle_seconds: float = STATS_SETTLE_SECONDS
    plot_dpi: int = PLOT_DPI

    def endpoints(self) -> list:
        all_endpoints = build_endpoints(self.base_url, self.target_header)
        if not self.only:
            return all_endpoints
        selected = [e for e in all_endpoints if e.key == self.only or e.model == self.only]
        if not selected:
            valid = ", ".join(f"{e.key}/{e.model}" for e in all_endpoints)
            raise SystemExit(f"--only {self.only!r} matches no endpoint (choose one of: {valid})")
        return selected


def parse_args(argv=None) -> RunConfig:
    """Parse command-line flags into a RunConfig, defaulting to the constants."""
    parser = argparse.ArgumentParser(
        description="Triton staged concurrency (ramp-up) throughput tester.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-url", default=TRITON_BASE_URL, help="Triton server base URL")
    parser.add_argument(
        "--target-header",
        default=SAGEMAKER_TARGET_HEADER,
        help="SageMaker target-model header name; pass '' to send no routing header",
    )
    parser.add_argument("--start", type=int, default=START_WORKERS, help="starting connections per endpoint")
    parser.add_argument("--increment", type=int, default=WORKER_INCREMENT, help="connections added per stage")
    parser.add_argument("--max", type=int, default=MAX_WORKERS, help="maximum connections per endpoint")
    parser.add_argument("--duration", type=float, default=STAGE_DURATION_SECONDS, help="measured seconds per stage")
    parser.add_argument("--warmup", type=float, default=WARMUP_SECONDS, help="uncounted seconds per stage")
    parser.add_argument("--plateau", type=float, default=PLATEAU_THRESHOLD, help="aggregate gain below this is flat")
    parser.add_argument("--plateau-stages", type=int, default=PLATEAU_STAGES_REQUIRED, help="consecutive flat stages to stop")
    parser.add_argument("--failure-stop", type=float, default=FAILURE_STOP_RATE, help="failure rate that stops the ramp")
    parser.add_argument("--min-requests", type=int, default=MIN_REQUESTS_PER_STAGE, help="ignore plateau below this many requests")
    parser.add_argument("--request-timeout", type=float, default=REQUEST_TIMEOUT_SECONDS, help="per-request timeout in seconds")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="directory for CSV, PNGs and run.json")
    parser.add_argument("--label", default="", help="free-form label recorded in run.json")
    parser.add_argument(
        "--only",
        default="",
        help="benchmark a single endpoint by key (onnx/python) or model name; default is all",
    )
    parser.add_argument(
        "--no-stats",
        dest="collect_stats",
        action="store_false",
        help="skip sampling Triton's /stats endpoint",
    )
    parser.add_argument(
        "--stats-settle",
        type=float,
        default=STATS_SETTLE_SECONDS,
        help="seconds to wait after a stage so Triton's async stats catch up",
    )
    args = parser.parse_args(argv)

    return RunConfig(
        base_url=args.base_url.rstrip("/"),
        target_header=args.target_header,
        start_workers=args.start,
        worker_increment=args.increment,
        max_workers=args.max,
        stage_duration=args.duration,
        warmup=args.warmup,
        plateau_threshold=args.plateau,
        plateau_stages=args.plateau_stages,
        failure_stop_rate=args.failure_stop,
        min_requests_per_stage=args.min_requests,
        request_timeout=args.request_timeout,
        output_dir=args.output_dir,
        label=args.label,
        only=args.only,
        collect_stats=args.collect_stats,
        stats_settle_seconds=args.stats_settle,
    )


# =============================================================================
# == INTERNALS ===============================================================
# =============================================================================


@dataclass
class StageState:
    """Handoff between the ramp controller and the long-lived worker tasks.

    Workers live across stages so their keep-alive connections stay warm. The
    controller bumps ``generation`` to announce a new stage; each worker notices
    and switches to that stage's counters.

    A request is attributed to a stage when it *completes* inside
    ``[measure_start, measure_deadline)``, which keeps requests/second an exact
    count over the measured window. During warmup the window is empty.
    """

    generation: int = 0
    deadline: float = math.inf
    measure_start: float = math.inf
    measure_deadline: float = math.inf
    stopping: bool = False


@dataclass
class Counters:
    """Raw per-endpoint, per-stage observations."""

    latencies: list = field(default_factory=list)
    successes: int = 0
    failures: int = 0
    errors: Counter = field(default_factory=Counter)


@dataclass
class StageMetrics:
    """Aggregated numbers for one endpoint at one concurrency level."""

    workers: int
    duration: float
    successes: int
    failures: int
    rps: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    failure_rate: float
    errors: Counter


def build_stage_schedule(start: int, increment: int, maximum: int) -> list:
    """Concurrency levels to visit: start, start+inc, ... then exactly maximum.

    The final step is shortened so the ramp always lands on ``maximum`` rather
    than overshooting it (1, 6, ... 96, 100 for the default constants).

    ``REFERENCE_CONCURRENCIES`` are folded in wherever the step would skip
    over them, so every run reports the secondary tiebreaker metric no matter
    what increment was chosen. The schedule stays deterministic, which is what
    makes experiments comparable.
    """
    if increment <= 0:
        raise ValueError("WORKER_INCREMENT must be positive")
    if maximum < start:
        raise ValueError("MAX_WORKERS must be >= START_WORKERS")

    schedule = []
    level = start
    while level < maximum:
        schedule.append(level)
        level += increment
    schedule.append(maximum)

    wanted = set(schedule)
    wanted.update(
        level for level in REFERENCE_CONCURRENCIES if start <= level <= maximum
    )
    return sorted(wanted)


def percentile(sorted_values: list, pct: float) -> float:
    """Linear-interpolated percentile over an already-sorted list."""
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * (pct / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return sorted_values[int(rank)]
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (rank - low)


def summarise(workers: int, duration: float, counters: Counters) -> StageMetrics:
    """Turn one stage's raw counters into the metrics used for plots."""
    total = counters.successes + counters.failures
    latencies = sorted(counters.latencies)
    return StageMetrics(
        workers=workers,
        duration=duration,
        successes=counters.successes,
        failures=counters.failures,
        rps=counters.successes / duration if duration > 0 else 0.0,
        mean_ms=statistics.fmean(latencies) if latencies else float("nan"),
        p50_ms=percentile(latencies, 50),
        p95_ms=percentile(latencies, 95),
        p99_ms=percentile(latencies, 99),
        max_ms=latencies[-1] if latencies else float("nan"),
        failure_rate=(counters.failures / total) if total else 0.0,
        errors=counters.errors,
    )


# --- Triton server-side statistics -------------------------------------------

async def fetch_model_stats(session: aiohttp.ClientSession, base_url: str, model: str):
    """Snapshot one model's Triton stats, or None when unavailable.

    Triton wraps the payload in a ``model_stats`` array, so unwrap the single
    entry for the model we asked about.
    """
    try:
        async with session.get(f"{base_url}/v2/models/{model}/stats") as response:
            if response.status != 200:
                return None
            body = await response.json()
    except Exception:
        return None

    entries = body.get("model_stats")
    if isinstance(entries, list):
        for entry in entries:
            if entry.get("name") == model:
                return entry
        return entries[0] if entries else None
    return body


def _counter(node: dict, key: str) -> dict:
    child = (node or {}).get(key) or {}
    return {"count": int(child.get("count", 0)), "ns": int(child.get("ns", 0))}


def _ns_delta(before: dict, after: dict, path: str) -> int:
    return int(after.get(path, 0)) - int(before.get(path, 0))


def diff_model_stats(before, after) -> dict:
    """Per-model server-side deltas between two /stats snapshots.

    Triton stats are cumulative since the model was loaded and there is no
    reset API, so every number here is an after-minus-before difference. Stats
    are also written *after* the response is returned, so the caller must let
    the snapshot settle before diffing.
    """
    if before is None or after is None:
        return {"available": False}

    inf_before = before.get("inference_stats", {})
    inf_after = after.get("inference_stats", {})

    def delta_of(key: str) -> dict:
        b = _counter(inf_before, key)
        a = _counter(inf_after, key)
        return {"count": a["count"] - b["count"], "ns": a["ns"] - b["ns"]}

    inference_count = int(after.get("inference_count", 0)) - int(before.get("inference_count", 0))
    execution_count = int(after.get("execution_count", 0)) - int(before.get("execution_count", 0))

    # batch_stats lives inside each compute_* section, keyed by batch size.
    batch_before = {b["batch_size"]: b for b in (before.get("batch_stats") or [])}
    batch_after = {b["batch_size"]: b for b in (after.get("batch_stats") or [])}
    batch_deltas = {}
    for size in sorted(set(batch_before) | set(batch_after)):
        b = batch_before.get(size, {})
        a = batch_after.get(size, {})
        count = int(((a.get("compute_infer") or {}).get("count", 0))) - int(
            ((b.get("compute_infer") or {}).get("count", 0))
        )
        if count:
            batch_deltas[str(size)] = count

    queue = delta_of("queue")
    compute = delta_of("compute_infer")
    return {
        "available": True,
        "success": delta_of("success"),
        "queue": queue,
        "compute_infer": compute,
        "compute_input": delta_of("compute_input"),
        "compute_output": delta_of("compute_output"),
        "inference_count": inference_count,
        "execution_count": execution_count,
        "avg_batch_size": (inference_count / execution_count) if execution_count else None,
        "batch_distribution": batch_deltas,
    }


async def settle_stats(session, cfg: RunConfig, model: str, before, timeout: float = 10.0):
    """Re-snapshot stats until Triton's async bookkeeping catches up.

    Triton writes request stats after the response is returned, so an immediate
    snapshot under-counts. Retry until the success count stops moving (Triton's
    own QA retries 10 x 1s; we poll faster within the same budget).
    """
    after = await fetch_model_stats(session, cfg.base_url, model)
    deadline = time.perf_counter() + timeout
    stable_for = 0
    while time.perf_counter() < deadline:
        await asyncio.sleep(0.25)
        latest = await fetch_model_stats(session, cfg.base_url, model)
        if latest is None:
            break
        prev_count = ((after or {}).get("inference_stats") or {}).get("success", {}).get("count")
        new_count = ((latest or {}).get("inference_stats") or {}).get("success", {}).get("count")
        if new_count == prev_count:
            stable_for += 1
            if stable_for >= 2:
                after = latest
                break
        else:
            stable_for = 0
        after = latest
    return after


async def fire_one(
    session: aiohttp.ClientSession, endpoint: Endpoint, counters: Counters, state: StageState
) -> None:
    """Issue a single inference request and record latency or the failure.

    Counted only when the response lands inside the active measurement window.
    """
    start = time.perf_counter()
    try:
        async with session.post(endpoint.url, json=endpoint.payload, headers=endpoint.headers) as response:
            # Drain the body so the connection returns to the keep-alive pool.
            await response.read()
            finished = time.perf_counter()
            counted = state.measure_start <= finished < state.measure_deadline
            if counted:
                if response.status == 200:
                    counters.successes += 1
                    counters.latencies.append((finished - start) * 1000.0)
                else:
                    counters.failures += 1
                    counters.errors[f"HTTP {response.status}"] += 1
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # timeouts, resets, DNS, connection refused...
        finished = time.perf_counter()
        if state.measure_start <= finished < state.measure_deadline:
            counters.failures += 1
            counters.errors[type(exc).__name__] += 1


async def worker(session: aiohttp.ClientSession, endpoint: Endpoint, state: StageState, buckets: dict) -> None:
    """One persistent connection driver: runs for the whole ramp, stage after stage."""
    seen_generation = 0
    counters = None
    while not state.stopping:
        if state.generation != seen_generation:
            seen_generation = state.generation
            counters = buckets[seen_generation][endpoint.key]

        if counters is None or time.perf_counter() >= state.deadline:
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            continue

        await fire_one(session, endpoint, counters, state)


def print_stage_table(stage_rows: list, endpoints: list) -> None:
    """Render the accumulated per-stage metrics as a fixed-width report."""
    header = (
        f"{'Workers':>7} | {'Endpoint':<26} | {'OK':>6} | {'Fail':>5} | {'Fail%':>6} | "
        f"{'RPS':>8} | {'Mean ms':>8} | {'p50 ms':>8} | {'p95 ms':>8} | {'p99 ms':>8} | {'AvgBS':>6}"
    )
    print("\n" + header)
    print("-" * len(header))
    for workers, metrics_by_key, aggregate_rps, stats_by_key in stage_rows:
        for endpoint in endpoints:
            m = metrics_by_key[endpoint.key]
            stats = stats_by_key.get(endpoint.key) or {}
            avg_bs = stats.get("avg_batch_size")
            avg_bs_txt = f"{avg_bs:.2f}" if isinstance(avg_bs, (int, float)) else "-"
            print(
                f"{workers:>7} | {endpoint.label:<26} | {m.successes:>6} | {m.failures:>5} | "
                f"{m.failure_rate * 100:>5.1f}% | {m.rps:>8.1f} | {m.mean_ms:>8.1f} | "
                f"{m.p50_ms:>8.1f} | {m.p95_ms:>8.1f} | {m.p99_ms:>8.1f} | {avg_bs_txt:>6}"
            )
        print(f"{workers:>7} | {'AGGREGATE':<26} | {'':>6} | {'':>5} | {'':>6} | {aggregate_rps:>8.1f}")
        print("-" * len(header))


def write_csv(path: Path, stage_rows: list, endpoints: list) -> None:
    """Persist every stage/endpoint measurement for later analysis.

    Includes the Triton server-side deltas (realised batch size, queue vs
    compute time) which are the actual subject of the batching experiments.
    """
    fields = [
        "workers", "endpoint", "duration_s", "successes", "failures", "failure_rate",
        "requests_per_second", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms", "top_errors",
        "triton_inference_count", "triton_execution_count", "avg_batch_size",
        "queue_ns", "compute_infer_ns", "batch_distribution",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for workers, metrics_by_key, aggregate_rps, stats_by_key in stage_rows:
            for endpoint in endpoints:
                m = metrics_by_key[endpoint.key]
                stats = stats_by_key.get(endpoint.key) or {}
                top_errors = "; ".join(f"{name} x{count}" for name, count in m.errors.most_common(3))
                batch_dist = "; ".join(
                    f"bs{k}=v{v}" for k, v in sorted(stats.get("batch_distribution", {}).items(), key=lambda kv: int(kv[0]))
                )
                writer.writerow(
                    {
                        "workers": workers,
                        "endpoint": endpoint.key,
                        "duration_s": f"{m.duration:.3f}",
                        "successes": m.successes,
                        "failures": m.failures,
                        "failure_rate": f"{m.failure_rate:.4f}",
                        "requests_per_second": f"{m.rps:.3f}",
                        "mean_ms": f"{m.mean_ms:.3f}",
                        "p50_ms": f"{m.p50_ms:.3f}",
                        "p95_ms": f"{m.p95_ms:.3f}",
                        "p99_ms": f"{m.p99_ms:.3f}",
                        "max_ms": f"{m.max_ms:.3f}",
                        "top_errors": top_errors,
                        "triton_inference_count": stats.get("inference_count", ""),
                        "triton_execution_count": stats.get("execution_count", ""),
                        "avg_batch_size": (
                            f"{stats['avg_batch_size']:.3f}" if stats.get("avg_batch_size") else ""
                        ),
                        "queue_ns": (stats.get("queue") or {}).get("ns", ""),
                        "compute_infer_ns": (stats.get("compute_infer") or {}).get("ns", ""),
                        "batch_distribution": batch_dist,
                    }
                )
            writer.writerow(
                {
                    "workers": workers,
                    "endpoint": "aggregate",
                    "duration_s": "",
                    "successes": "",
                    "failures": "",
                    "failure_rate": "",
                    "requests_per_second": f"{aggregate_rps:.3f}",
                    "mean_ms": "", "p50_ms": "", "p95_ms": "", "p99_ms": "", "max_ms": "",
                    "top_errors": "",
                }
            )


def plot_results(stage_rows: list, stop_reason: str, output_dir: Path, endpoints: list, dpi: int = PLOT_DPI) -> list:
    """Draw throughput / latency / error line graphs and save them as PNGs."""
    if not stage_rows:
        print("No stages completed, skipping graphs.")
        return []

    workers = [row[0] for row in stage_rows]
    per_endpoint = {
        endpoint.key: {
            "rps": [row[1][endpoint.key].rps for row in stage_rows],
            "mean": [row[1][endpoint.key].mean_ms for row in stage_rows],
            "p95": [row[1][endpoint.key].p95_ms for row in stage_rows],
            "fail": [row[1][endpoint.key].failure_rate * 100 for row in stage_rows],
            "avg_bs": [
                (row[3].get(endpoint.key) or {}).get("avg_batch_size") for row in stage_rows
            ],
        }
        for endpoint in endpoints
    }
    aggregate_rps = [row[2] for row in stage_rows]
    saved = []

    def finish(fig, filename: str) -> None:
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        saved.append(path)

    def annotate_stop(ax) -> None:
        if stop_reason:
            ax.axvline(workers[-1], color="red", linestyle=":", linewidth=1, alpha=0.8)
            ax.annotate(
                "ramp stopped",
                xy=(workers[-1], ax.get_ylim()[1]),
                xytext=(-6, -12),
                textcoords="offset points",
                ha="right",
                va="top",
                color="red",
                fontsize=8,
            )

    # --- Throughput ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    for endpoint in endpoints:
        ax.plot(workers, per_endpoint[endpoint.key]["rps"], marker="o", linewidth=1.8, label=endpoint.label)
    ax.plot(workers, aggregate_rps, marker="s", linewidth=2.2, linestyle="--", color="black", label="Aggregate")
    ax.set_xlabel("Concurrent connections per endpoint")
    ax.set_ylabel("Successful requests / second")
    ax.set_title("Triton ramp-up: throughput vs concurrency")
    ax.grid(True, alpha=0.3)
    ax.legend()
    annotate_stop(ax)
    finish(fig, "ramp-throughput.png")

    # --- Latency -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for endpoint in endpoints:
        axes[0].plot(workers, per_endpoint[endpoint.key]["mean"], marker="o", linewidth=1.8, label=endpoint.label)
        axes[1].plot(workers, per_endpoint[endpoint.key]["p95"], marker="o", linewidth=1.8, label=endpoint.label)
    axes[0].set_title("Mean latency vs concurrency")
    axes[0].set_ylabel("Milliseconds")
    axes[1].set_title("p95 latency vs concurrency")
    axes[1].set_ylabel("Milliseconds")
    for ax in axes:
        ax.set_xlabel("Concurrent connections per endpoint")
        ax.grid(True, alpha=0.3)
        ax.legend()
        annotate_stop(ax)
    finish(fig, "ramp-latency.png")

    # --- Failure rate --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    for endpoint in endpoints:
        ax.plot(workers, per_endpoint[endpoint.key]["fail"], marker="o", linewidth=1.8, label=endpoint.label)
    ax.axhline(FAILURE_STOP_RATE * 100, color="red", linestyle="--", linewidth=1, label=f"Stop threshold ({FAILURE_STOP_RATE * 100:.0f}%)")
    ax.set_xlabel("Concurrent connections per endpoint")
    ax.set_ylabel("Failed requests (%)")
    ax.set_title("Triton ramp-up: failure rate vs concurrency")
    ax.grid(True, alpha=0.3)
    ax.legend()
    annotate_stop(ax)
    finish(fig, "ramp-errors.png")

    # --- Realised batch size (server-side, the batching payoff) --------------
    if any(any(v is not None for v in per_endpoint[e.key]["avg_bs"]) for e in endpoints):
        fig, ax = plt.subplots(figsize=(9, 5))
        for endpoint in endpoints:
            ax.plot(workers, per_endpoint[endpoint.key]["avg_bs"], marker="o", linewidth=1.8, label=endpoint.label)
        ax.set_xlabel("Concurrent connections per endpoint")
        ax.set_ylabel("Realised avg batch size (inference/execution)")
        ax.set_title("Triton ramp-up: dynamic batching realised batch size")
        ax.grid(True, alpha=0.3)
        ax.legend()
        annotate_stop(ax)
        finish(fig, "ramp-batch-size.png")

    # --- Combined overview ---------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].plot(workers, aggregate_rps, marker="s", color="black", linewidth=2, label="Aggregate")
    axes[0, 0].set_title("Aggregate throughput")
    axes[0, 0].set_ylabel("Requests / second")
    for endpoint in endpoints:
        axes[0, 1].plot(workers, per_endpoint[endpoint.key]["rps"], marker="o", linewidth=1.8, label=endpoint.label)
        axes[1, 0].plot(workers, per_endpoint[endpoint.key]["mean"], marker="o", linewidth=1.8, label=endpoint.label)
        axes[1, 1].plot(workers, per_endpoint[endpoint.key]["fail"], marker="o", linewidth=1.8, label=endpoint.label)
    axes[0, 1].set_title("Per-endpoint throughput")
    axes[0, 1].set_ylabel("Requests / second")
    axes[1, 0].set_title("Mean latency")
    axes[1, 0].set_ylabel("Milliseconds")
    axes[1, 1].set_title("Failure rate")
    axes[1, 1].set_ylabel("Percent")
    for ax in axes.flat:
        ax.set_xlabel("Concurrent connections per endpoint")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        annotate_stop(ax)
    fig.suptitle("Triton ramp-up load test", y=0.995)
    finish(fig, "ramp-summary.png")

    return saved


async def run_ramp(cfg: RunConfig) -> tuple:
    """Drive the staged ramp and return (stage rows, stop reason).

    Each stage row is ``(workers, metrics_by_endpoint, aggregate_rps,
    triton_stats_by_endpoint)``. The last entry holds server-side deltas from
    Triton's ``/stats`` and is ``{"available": False}`` when stats are off.
    """
    endpoints = cfg.endpoints()
    schedule = build_stage_schedule(cfg.start_workers, cfg.worker_increment, cfg.max_workers)
    total_connections = cfg.max_workers * len(endpoints)

    connector = aiohttp.TCPConnector(
        limit=total_connections + 16,
        limit_per_host=total_connections + 16,  # both endpoints share one host
        keepalive_timeout=cfg.keepalive_timeout,
        ttl_dns_cache=300,
    )
    timeout = aiohttp.ClientTimeout(total=cfg.request_timeout)

    state = StageState()
    buckets = {}
    stage_rows = []
    stop_reason = ""
    workers_by_endpoint = {endpoint.key: 0 for endpoint in endpoints}
    previous_aggregate_rps = None
    flat_stages = 0

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        try:
            for index, level in enumerate(schedule, start=1):
                buckets[index] = {endpoint.key: Counters() for endpoint in endpoints}

                # Snapshot server-side counters before the stage so the diff
                # covers exactly this stage's work.
                stats_before = {}
                if cfg.collect_stats:
                    for endpoint in endpoints:
                        stats_before[endpoint.key] = await fetch_model_stats(
                            session, cfg.base_url, endpoint.model
                        )

                # Grow the pool to the next level; existing workers keep running
                # so their connections stay warm between stages.
                for endpoint in endpoints:
                    while workers_by_endpoint[endpoint.key] < level:
                        tasks.append(asyncio.create_task(worker(session, endpoint, state, buckets)))
                        workers_by_endpoint[endpoint.key] += 1

                # Warmup: load is applied but the measurement window is closed,
                # so connections get established without skewing the numbers.
                state.measure_start = math.inf
                state.measure_deadline = math.inf
                now = time.perf_counter()
                warmup_until = now + cfg.warmup
                measure_start = warmup_until
                measure_deadline = measure_start + cfg.stage_duration
                state.deadline = measure_deadline
                state.generation = index

                while time.perf_counter() < warmup_until:
                    await asyncio.sleep(0.02)

                state.measure_start = measure_start
                state.measure_deadline = measure_deadline
                while time.perf_counter() < measure_deadline:
                    await asyncio.sleep(0.05)

                # Close the window before reading counters so late responses that
                # started inside the window are not attributed to this stage.
                state.measure_start = math.inf
                state.measure_deadline = math.inf
                duration = measure_deadline - measure_start

                metrics = {
                    endpoint.key: summarise(level, duration, buckets[index][endpoint.key])
                    for endpoint in endpoints
                }
                aggregate_rps = sum(metrics[endpoint.key].rps for endpoint in endpoints)

                # Triton writes request stats after the response is returned, so
                # wait for the numbers to settle before taking the "after" shot.
                stats_by_key = {}
                if cfg.collect_stats:
                    if cfg.stats_settle_seconds > 0:
                        await asyncio.sleep(cfg.stats_settle_seconds)
                    for endpoint in endpoints:
                        after = await settle_stats(
                            session, cfg, endpoint.model, stats_before.get(endpoint.key)
                        )
                        stats_by_key[endpoint.key] = diff_model_stats(
                            stats_before.get(endpoint.key), after
                        )
                else:
                    stats_by_key = {endpoint.key: {"available": False} for endpoint in endpoints}

                stage_rows.append((level, metrics, aggregate_rps, stats_by_key))

                total_failures = sum(metrics[endpoint.key].failures for endpoint in endpoints)
                total_requests = total_failures + sum(metrics[endpoint.key].successes for endpoint in endpoints)
                stage_failure_rate = total_failures / total_requests if total_requests else 0.0

                batch_txt = ""
                if cfg.collect_stats:
                    sizes = [
                        (stats_by_key[e.key] or {}).get("avg_batch_size") for e in endpoints
                    ]
                    sizes = [s for s in sizes if isinstance(s, (int, float))]
                    if sizes:
                        batch_txt = f" | avg batch {statistics.fmean(sizes):.2f}"
                print(
                    f"[stage {index:>3}] {level:>4} conn/endpoint | aggregate {aggregate_rps:7.1f} rps | "
                    f"failures {stage_failure_rate * 100:5.1f}%{batch_txt}"
                )

                # --- Early-stop rule 1: too many failures ---------------------
                if stage_failure_rate >= cfg.failure_stop_rate:
                    stop_reason = (
                        f"failure rate {stage_failure_rate * 100:.1f}% reached the "
                        f"{cfg.failure_stop_rate * 100:.0f}% limit at {level} connections/endpoint"
                    )
                    break

                # --- Early-stop rule 2: throughput plateau --------------------
                total_requests_ok = total_requests - total_failures
                if (
                    previous_aggregate_rps is not None
                    and previous_aggregate_rps > 0
                    and total_requests_ok >= cfg.min_requests_per_stage
                ):
                    gain = (aggregate_rps - previous_aggregate_rps) / previous_aggregate_rps
                    if gain < cfg.plateau_threshold:
                        flat_stages += 1
                        if flat_stages >= cfg.plateau_stages:
                            stop_reason = (
                                f"aggregate throughput stopped improving "
                                f"(<{cfg.plateau_threshold * 100:.0f}% for {flat_stages} consecutive stages) "
                                f"at {level} connections/endpoint"
                            )
                            break
                    else:
                        flat_stages = 0
                previous_aggregate_rps = aggregate_rps
        finally:
            state.stopping = True
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    return stage_rows, stop_reason


def git_sha() -> str:
    """Current commit SHA of this repo, or "" when not available."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def rps_at_reference_levels(stage_rows: list, endpoints: list) -> dict:
    """Aggregate RPS at each reference concurrency, when that stage ran."""
    levels = {}
    for workers, metrics_by_key, aggregate_rps, _stats in stage_rows:
        levels[workers] = round(aggregate_rps, 3)
    return {
        str(level): levels.get(level) for level in REFERENCE_CONCURRENCIES
    }


def write_stats_json(path: Path, stage_rows: list, endpoints: list) -> None:
    """Persist the raw Triton stats deltas per stage for offline debugging."""
    payload = [
        {
            "workers": workers,
            "endpoints": {
                endpoint.key: stats_by_key.get(endpoint.key, {}) for endpoint in endpoints
            },
        }
        for workers, _metrics, _aggregate, stats_by_key in stage_rows
    ]
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_run_json(
    path: Path, cfg: RunConfig, stage_rows: list, stop_reason: str, started_at, elapsed: float
) -> None:
    """Record the full provenance of one run so results stay reproducible."""
    endpoints = cfg.endpoints()
    peak = max(stage_rows, key=lambda row: row[2]) if stage_rows else None
    per_endpoint_peak = {}
    if peak:
        for endpoint in endpoints:
            best = max(stage_rows, key=lambda row: row[1][endpoint.key].rps)
            per_endpoint_peak[endpoint.key] = {
                "model": endpoint.model,
                "peak_rps": round(best[1][endpoint.key].rps, 3),
                "workers": best[0],
                "p50_ms": round(best[1][endpoint.key].p50_ms, 3),
                "p99_ms": round(best[1][endpoint.key].p99_ms, 3),
                "avg_batch_size": (best[3].get(endpoint.key) or {}).get("avg_batch_size"),
            }

    payload = {
        "schema_version": 1,
        "label": cfg.label,
        "started_at": started_at.isoformat(),
        "finished_at": (started_at + timedelta(seconds=elapsed)).isoformat(),
        "elapsed_seconds": round(elapsed, 3),
        "git_sha": git_sha(),
        "run": {
            "base_url": cfg.base_url,
            "schedule": [row[0] for row in stage_rows],
            "stage_duration_seconds": cfg.stage_duration,
            "warmup_seconds": cfg.warmup,
            "start_workers": cfg.start_workers,
            "worker_increment": cfg.worker_increment,
            "max_workers": cfg.max_workers,
            "thresholds": {
                "plateau_threshold": cfg.plateau_threshold,
                "plateau_stages_required": cfg.plateau_stages,
                "failure_stop_rate": cfg.failure_stop_rate,
                "min_requests_per_stage": cfg.min_requests_per_stage,
            },
        },
        "endpoints": [
            {"key": e.key, "model": e.model, "url": e.url, "headers": e.headers} for e in endpoints
        ],
        "result": {
            "stages_measured": len(stage_rows),
            "stop_reason": stop_reason or "reached MAX_WORKERS without hitting a stop rule",
            "peak_aggregate_rps": round(peak[2], 3) if peak else None,
            "peak_workers": peak[0] if peak else None,
            "rps_at_reference_concurrency": rps_at_reference_levels(stage_rows, endpoints),
            "per_endpoint_peak": per_endpoint_peak,
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main(argv=None) -> int:
    # Ramp tests are often left running with output redirected to a log file,
    # where Python block-buffers stdout. Line buffering keeps that log live.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)
    cfg = parse_args(argv)
    endpoints = cfg.endpoints()
    schedule = build_stage_schedule(cfg.start_workers, cfg.worker_increment, cfg.max_workers)

    print("================ TRITON RAMP-UP LOAD TEST ================")
    print(f"Target          : {cfg.base_url}")
    for endpoint in endpoints:
        print(f"  endpoint      : {endpoint.label} -> {endpoint.url}")
    print(f"Ramp            : {schedule[0]} -> {schedule[-1]} connections/endpoint, +{cfg.worker_increment} per stage")
    print(f"Stages          : {len(schedule)} x {cfg.stage_duration:.1f}s "
          f"(+{cfg.warmup:.1f}s warmup each)")
    print(f"Early stop      : failure rate >= {cfg.failure_stop_rate * 100:.0f}%, or aggregate gain "
          f"< {cfg.plateau_threshold * 100:.0f}% for {cfg.plateau_stages} stages")
    print(f"Server stats    : {'on' if cfg.collect_stats else 'off'}")
    if cfg.only:
        print(f"Single endpoint : {cfg.only}")
    if cfg.label:
        print(f"Label           : {cfg.label}")
    print("==========================================================\n")

    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    try:
        stage_rows, stop_reason = asyncio.run(run_ramp(cfg))
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130
    elapsed = time.perf_counter() - started

    if not stage_rows:
        print("No stages were measured — nothing to report.")
        return 1

    print_stage_table(stage_rows, endpoints)
    print(f"\nRamp finished in {elapsed:.1f}s after {len(stage_rows)} stage(s).")
    print(f"Stop reason: {stop_reason or 'reached MAX_WORKERS without hitting a stop rule'}")

    peak = max(stage_rows, key=lambda row: row[2])
    print(f"Peak aggregate throughput: {peak[2]:.1f} rps at {peak[0]} connections/endpoint")

    # Never clobber an earlier run: if the directory already holds results,
    # keep both by moving this one into a timestamped subdirectory.
    output_dir = cfg.output_dir
    if (output_dir / CSV_NAME).exists() or (output_dir / "run.json").exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = output_dir / stamp
        print(f"(output directory already populated — writing to {output_dir})")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / CSV_NAME
    write_csv(csv_path, stage_rows, endpoints)
    print(f"\nCSV  : {csv_path}")

    for path in plot_results(stage_rows, stop_reason, output_dir, endpoints, cfg.plot_dpi):
        print(f"PNG  : {path}")

    run_json_path = output_dir / "run.json"
    write_run_json(run_json_path, cfg, stage_rows, stop_reason, started_at, elapsed)
    print(f"JSON : {run_json_path}")

    if cfg.collect_stats:
        stats_path = output_dir / "stats.deltas.json"
        write_stats_json(stats_path, stage_rows, endpoints)
        print(f"STATS: {stats_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
