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

Dependencies: aiohttp, matplotlib
"""

import asyncio
import csv
import math
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
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

# --- CLIENT BEHAVIOUR --------------------------------------------------------
REQUEST_TIMEOUT_SECONDS = 30.0
KEEPALIVE_TIMEOUT_SECONDS = 75.0
POLL_INTERVAL_SECONDS = 0.01    # how often an idle worker checks for a new stage

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
    url: str
    payload: dict
    headers: dict = field(default_factory=dict)


ENDPOINTS = [
    Endpoint(
        key="onnx",
        label="miniLM-onnx (Priority 1)",
        url=f"{TRITON_BASE_URL}/v2/models/miniLM-onnx/infer",
        payload=PAYLOAD_ONNX,
        headers={SAGEMAKER_TARGET_HEADER: "miniLM-onnx"},
    ),
    Endpoint(
        key="python",
        label="miniLM python (Priority 2)",
        url=f"{TRITON_BASE_URL}/v2/models/miniLM/infer",
        payload=PAYLOAD_PYTHON,
    ),
]

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
    return schedule


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


def print_stage_table(stage_rows: list) -> None:
    """Render the accumulated per-stage metrics as a fixed-width report."""
    header = (
        f"{'Workers':>7} | {'Endpoint':<26} | {'OK':>6} | {'Fail':>5} | {'Fail%':>6} | "
        f"{'RPS':>8} | {'Mean ms':>8} | {'p50 ms':>8} | {'p95 ms':>8} | {'p99 ms':>8}"
    )
    print("\n" + header)
    print("-" * len(header))
    for workers, metrics_by_key, aggregate_rps in stage_rows:
        for endpoint in ENDPOINTS:
            m = metrics_by_key[endpoint.key]
            print(
                f"{workers:>7} | {endpoint.label:<26} | {m.successes:>6} | {m.failures:>5} | "
                f"{m.failure_rate * 100:>5.1f}% | {m.rps:>8.1f} | {m.mean_ms:>8.1f} | "
                f"{m.p50_ms:>8.1f} | {m.p95_ms:>8.1f} | {m.p99_ms:>8.1f}"
            )
        print(f"{workers:>7} | {'AGGREGATE':<26} | {'':>6} | {'':>5} | {'':>6} | {aggregate_rps:>8.1f}")
        print("-" * len(header))


def write_csv(path: Path, stage_rows: list) -> None:
    """Persist every stage/endpoint measurement for later analysis."""
    fields = [
        "workers", "endpoint", "duration_s", "successes", "failures", "failure_rate",
        "requests_per_second", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms", "top_errors",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for workers, metrics_by_key, aggregate_rps in stage_rows:
            for endpoint in ENDPOINTS:
                m = metrics_by_key[endpoint.key]
                top_errors = "; ".join(f"{name} x{count}" for name, count in m.errors.most_common(3))
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


def plot_results(stage_rows: list, stop_reason: str, output_dir: Path) -> list:
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
        }
        for endpoint in ENDPOINTS
    }
    aggregate_rps = [row[2] for row in stage_rows]
    saved = []

    def finish(fig, filename: str) -> None:
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=PLOT_DPI)
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
    for endpoint in ENDPOINTS:
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
    for endpoint in ENDPOINTS:
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
    for endpoint in ENDPOINTS:
        ax.plot(workers, per_endpoint[endpoint.key]["fail"], marker="o", linewidth=1.8, label=endpoint.label)
    ax.axhline(FAILURE_STOP_RATE * 100, color="red", linestyle="--", linewidth=1, label=f"Stop threshold ({FAILURE_STOP_RATE * 100:.0f}%)")
    ax.set_xlabel("Concurrent connections per endpoint")
    ax.set_ylabel("Failed requests (%)")
    ax.set_title("Triton ramp-up: failure rate vs concurrency")
    ax.grid(True, alpha=0.3)
    ax.legend()
    annotate_stop(ax)
    finish(fig, "ramp-errors.png")

    # --- Combined overview ---------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].plot(workers, aggregate_rps, marker="s", color="black", linewidth=2, label="Aggregate")
    axes[0, 0].set_title("Aggregate throughput")
    axes[0, 0].set_ylabel("Requests / second")
    for endpoint in ENDPOINTS:
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


async def run_ramp() -> tuple:
    """Drive the staged ramp and return (stage rows, stop reason)."""
    schedule = build_stage_schedule(START_WORKERS, WORKER_INCREMENT, MAX_WORKERS)
    total_connections = MAX_WORKERS * len(ENDPOINTS)

    connector = aiohttp.TCPConnector(
        limit=total_connections + 16,
        limit_per_host=total_connections + 16,  # both endpoints share one host
        keepalive_timeout=KEEPALIVE_TIMEOUT_SECONDS,
        ttl_dns_cache=300,
    )
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)

    state = StageState()
    buckets = {}
    stage_rows = []
    stop_reason = ""
    workers_by_endpoint = {endpoint.key: 0 for endpoint in ENDPOINTS}
    previous_aggregate_rps = None
    flat_stages = 0

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        try:
            for index, level in enumerate(schedule, start=1):
                buckets[index] = {endpoint.key: Counters() for endpoint in ENDPOINTS}

                # Grow the pool to the next level; existing workers keep running
                # so their connections stay warm between stages.
                for endpoint in ENDPOINTS:
                    while workers_by_endpoint[endpoint.key] < level:
                        tasks.append(asyncio.create_task(worker(session, endpoint, state, buckets)))
                        workers_by_endpoint[endpoint.key] += 1

                # Warmup: load is applied but the measurement window is closed,
                # so connections get established without skewing the numbers.
                state.measure_start = math.inf
                state.measure_deadline = math.inf
                now = time.perf_counter()
                warmup_until = now + WARMUP_SECONDS
                measure_start = warmup_until
                measure_deadline = measure_start + STAGE_DURATION_SECONDS
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
                    for endpoint in ENDPOINTS
                }
                aggregate_rps = sum(metrics[endpoint.key].rps for endpoint in ENDPOINTS)
                stage_rows.append((level, metrics, aggregate_rps))

                total_failures = sum(metrics[endpoint.key].failures for endpoint in ENDPOINTS)
                total_requests = total_failures + sum(metrics[endpoint.key].successes for endpoint in ENDPOINTS)
                stage_failure_rate = total_failures / total_requests if total_requests else 0.0

                print(
                    f"[stage {index:>3}] {level:>4} conn/endpoint | aggregate {aggregate_rps:7.1f} rps | "
                    f"failures {stage_failure_rate * 100:5.1f}%"
                )

                # --- Early-stop rule 1: too many failures ---------------------
                if stage_failure_rate >= FAILURE_STOP_RATE:
                    stop_reason = (
                        f"failure rate {stage_failure_rate * 100:.1f}% reached the "
                        f"{FAILURE_STOP_RATE * 100:.0f}% limit at {level} connections/endpoint"
                    )
                    break

                # --- Early-stop rule 2: throughput plateau --------------------
                total_requests_ok = total_requests - total_failures
                if (
                    previous_aggregate_rps is not None
                    and previous_aggregate_rps > 0
                    and total_requests_ok >= MIN_REQUESTS_PER_STAGE
                ):
                    gain = (aggregate_rps - previous_aggregate_rps) / previous_aggregate_rps
                    if gain < PLATEAU_THRESHOLD:
                        flat_stages += 1
                        if flat_stages >= PLATEAU_STAGES_REQUIRED:
                            stop_reason = (
                                f"aggregate throughput stopped improving "
                                f"(<{PLATEAU_THRESHOLD * 100:.0f}% for {flat_stages} consecutive stages) "
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


def main() -> int:
    schedule = build_stage_schedule(START_WORKERS, WORKER_INCREMENT, MAX_WORKERS)
    print("================ TRITON RAMP-UP LOAD TEST ================")
    print(f"Target          : {TRITON_BASE_URL}")
    for endpoint in ENDPOINTS:
        print(f"  endpoint      : {endpoint.label} -> {endpoint.url}")
    print(f"Ramp            : {schedule[0]} -> {schedule[-1]} connections/endpoint, +{WORKER_INCREMENT} per stage")
    print(f"Stages          : {len(schedule)} x {STAGE_DURATION_SECONDS:.1f}s "
          f"(+{WARMUP_SECONDS:.1f}s warmup each)")
    print(f"Early stop      : failure rate >= {FAILURE_STOP_RATE * 100:.0f}%, or aggregate gain "
          f"< {PLATEAU_THRESHOLD * 100:.0f}% for {PLATEAU_STAGES_REQUIRED} stages")
    print("==========================================================\n")

    started = time.perf_counter()
    try:
        stage_rows, stop_reason = asyncio.run(run_ramp())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130
    elapsed = time.perf_counter() - started

    if not stage_rows:
        print("No stages were measured — nothing to report.")
        return 1

    print_stage_table(stage_rows)
    print(f"\nRamp finished in {elapsed:.1f}s after {len(stage_rows)} stage(s).")
    print(f"Stop reason: {stop_reason or 'reached MAX_WORKERS without hitting a stop rule'}")

    peak = max(stage_rows, key=lambda row: row[2])
    print(f"Peak aggregate throughput: {peak[2]:.1f} rps at {peak[0]} connections/endpoint")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / CSV_NAME
    write_csv(csv_path, stage_rows)
    print(f"\nCSV  : {csv_path}")

    for path in plot_results(stage_rows, stop_reason, OUTPUT_DIR):
        print(f"PNG  : {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
