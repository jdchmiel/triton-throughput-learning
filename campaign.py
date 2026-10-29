"""
campaign.py — autonomous Triton configuration-search driver.

Copies the karpathy/autoresearch loop, with the roles mapped as:

    autoresearch            this campaign
    ---------------------   -------------------------------------------------
    prepare.py (fixed)      experiment_config.py + ramp-up.py (never mutated)
    train.py (mutated)      the model config JSON sent to Triton
    program.md              program.md (the human-readable protocol)
    fixed 5-min budget      fixed ramp schedule (same stages every experiment)
    val_bpb ratchet         peak aggregate requests/second
    results.tsv             experiments/<campaign>/results.tsv
    analysis.ipynb          analyze.py + progress.png

Each iteration: propose a config -> apply it to the live server -> verify it
took effect -> run an identical ramp-up load test -> record peak throughput ->
keep it only if it beats the incumbent. The campaign runs unattended until it
hits an experiment cap, a plateau count, or a wall-clock budget, then restores
the original configuration.

Run it with no arguments for a full campaign:

    python campaign.py

Useful controls:

    python campaign.py --max-experiments 6 --budget 900
    python campaign.py --model miniLM-onnx --dry-run
"""

import argparse
import csv
import io
import json
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import experiment_config as ec

# =============================================================================
# == CAMPAIGN DEFAULTS =======================================================
# =============================================================================

TRITON_BASE_URL = "http://192.168.1.151:8000"
TARGET_MODEL = "miniLM-onnx"          # which model's config is under test
RAMP_ENDPOINT_KEY = "onnx"            # matching ramp-up.py --only selector

EXPERIMENTS_ROOT = Path("experiments")
RAMP_SCRIPT = Path(__file__).parent / "ramp-up.py"
PYTHON = sys.executable

# --- ramp shape used by EVERY experiment (the fixed measurement budget) ------
RAMP_MAX_WORKERS = 60
RAMP_INCREMENT = 5
RAMP_DURATION = 8.0
RAMP_WARMUP = 1.0

# --- campaign stopping rules -------------------------------------------------
MAX_EXPERIMENTS = 40                  # hard cap on configs evaluated
PLATEAU_COUNT = 6                     # consecutive non-improvements -> stop
MAX_CONSECUTIVE_FAILURES = 4          # apply/ramp failures in a row -> stop
CAMPAIGN_BUDGET_SECONDS = 3 * 3600    # wall-clock ceiling
MIN_IMPROVEMENT = 0.02                # a "real" gain must beat this (2%)
RAMP_SUBPROCESS_TIMEOUT = 900         # per-experiment hang guard

RESULTS_TSV = "results.tsv"


# =============================================================================
# == RESULT RECORD ===========================================================
# =============================================================================


@dataclass
class ExperimentResult:
    number: int
    variant_label: str
    fingerprint: str
    peak_rps: float
    peak_workers: int
    folder: str
    status: str                      # ok | apply-failed | ramp-failed | crashed
    detail: str = ""
    avg_batch_size: Optional[float] = None
    p50_ms: Optional[float] = None
    p99_ms: Optional[float] = None
    stages: int = 0
    stop_reason: str = ""
    duration_seconds: float = 0.0
    rps_at: dict = field(default_factory=dict)
    diff: dict = field(default_factory=dict)


# =============================================================================
# == CAMPAIGN =================================================================
# =============================================================================


class Campaign:
    def __init__(self, args):
        self.args = args
        self.client = ec.TritonClient(args.base_url)
        self.model = args.model
        # Filled in by preflight(): the config to restore when the campaign ends.
        self.baseline: Optional[dict] = None
        self.baseline_fp: Optional[str] = None
        self.started_at = datetime.now(timezone.utc)
        self.started_monotonic = time.monotonic()

        stamp = self.started_at.strftime("%Y%m%d-%H%M%S")
        self.campaign_dir = Path(args.out) / stamp
        self.results = []
        self.best_rps = 0.0
        self.best_variant: Optional[ec.Variant] = None
        self.best_fingerprint: Optional[str] = None
        self.consecutive_plateaus = 0
        self.consecutive_failures = 0
        self.sweep_done = False          # plateau rule only applies during descent
        self.tried = {}              # fingerprint -> Variant
        self._restore_done = False

    # --- lifecycle -----------------------------------------------------------
    def install_signal_guard(self) -> None:
        """Restore the baseline config if the operator hits Ctrl-C."""
        def handler(signum, frame):
            print("\n[campaign] interrupted — restoring baseline configuration...")
            self.restore_baseline()
            print("[campaign] baseline restored. Exiting.")
            sys.exit(130)

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def restore_baseline(self) -> None:
        if self._restore_done or self.baseline is None:
            return
        self._restore_done = True
        if self.args.dry_run:
            print("[campaign] dry run: server untouched, no restore needed.")
            return
        try:
            self.client.restore(self.model, self.baseline)
            state, _ = self.client.model_state(self.model)
            print(f"[campaign] {self.model} restored to baseline (state={state}).")
        except Exception as exc:  # never let cleanup raise over the real error
            print(f"[campaign] WARNING: could not restore baseline: {exc}")

    def preflight(self) -> bool:
        """Confirm the server can actually be reconfigured at runtime."""
        print(f"[preflight] server: {json.dumps(self.client.server_metadata().get('name', '?'))} "
              f"version {self.client.server_metadata().get('version', '?')}")
        if not self.client.health_ready():
            print("[preflight] FAIL: /v2/health/ready is not OK — is Triton running?")
            return False

        probe = self.client.control_mode_probe()
        if not probe["load_api_available"]:
            if self.args.dry_run:
                print("[preflight] NOTE: load API unavailable, but --dry-run does not need it.")
            else:
                print("[preflight] FAIL: runtime model load/unload is not permitted.")
                print(f"            server said: {probe['error']}")
                print(
                    "\n            The campaign changes configs through the Triton load API, which\n"
                    "            requires the server to run with --model-control-mode=explicit.\n"
                    "            Update triton-models/run_triton.sh and restart the server, then\n"
                    "            re-run this campaign. See EXPERIMENT_CAMPAIGN_PLAN.md.\n"
                    "            (Use --dry-run to exercise the search without a server change.)"
                )
                return False

        try:
            self.baseline = self.client.get_config(self.model)
        except ec.TritonError as exc:
            print(f"[preflight] FAIL: cannot read config for {self.model}: {exc}")
            return False

        self.baseline_fp = ec.config_fingerprint(self.baseline)
        print(f"[preflight] OK: load API available; baseline fingerprint {self.baseline_fp}")
        print(f"[preflight] baseline max_batch_size={self.baseline.get('max_batch_size')}, "
              f"queue_delay={self.baseline.get('dynamic_batching', {}).get('max_queue_delay_microseconds')}us, "
              f"instance_count={(self.baseline.get('instance_group') or [{}])[0].get('count')}")
        return True

    # --- folder helpers ------------------------------------------------------
    def _require_baseline(self) -> dict:
        """The captured baseline config.

        Only available once preflight() has succeeded; every experiment, and
        the restore at the end, depends on that snapshot being present.
        """
        if self.baseline is None:
            raise RuntimeError("baseline config not captured — preflight did not complete")
        return self.baseline

    def staging_dir(self, number: int) -> Path:
        return self.campaign_dir / f"exp-{number:02d}_pending"

    def final_dir(self, number: int, peak_rps: float) -> Path:
        return self.campaign_dir / f"exp-{number:02d}_{int(round(peak_rps))}rps"

    # --- one experiment ------------------------------------------------------
    def run_experiment(self, number: int, variant: ec.Variant) -> ExperimentResult:
        baseline = self._require_baseline()
        candidate = ec.apply_variant(baseline, variant)
        fingerprint = ec.config_fingerprint(candidate)
        diff = ec.config_diff(baseline, candidate)
        staging = self.staging_dir(number)
        staging.mkdir(parents=True, exist_ok=True)

        (staging / "config.baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
        (staging / "config.sent.json").write_text(json.dumps(candidate, indent=2) + "\n")
        (staging / "config.diff.json").write_text(json.dumps(diff, indent=2) + "\n")

        started = time.monotonic()

        # --- apply -----------------------------------------------------------
        if self.args.dry_run:
            print(f"[exp {number:02d}] DRY RUN — would apply {variant.label()} ({fingerprint})")
            dry_dir = self.campaign_dir / f"exp-{number:02d}_dryrun"
            if dry_dir.exists():
                shutil.rmtree(dry_dir)
            staging.rename(dry_dir)
            return ExperimentResult(number, variant.label(), fingerprint, 0.0, 0, dry_dir.name,
                                    "dry-run", duration_seconds=time.monotonic() - started, diff=diff)

        try:
            applied = self.client.apply_config(self.model, candidate)
            (staging / "config.applied.json").write_text(json.dumps(applied, indent=2) + "\n")
        except ec.TritonError as exc:
            print(f"[exp {number:02d}] APPLY FAILED: {exc}")
            return ExperimentResult(number, variant.label(), fingerprint, 0.0, 0, staging.name,
                                    "apply-failed", str(exc),
                                    duration_seconds=time.monotonic() - started, diff=diff)

        # --- measure ---------------------------------------------------------
        cmd = [
            self.args.python, str(RAMP_SCRIPT),
            "--base-url", self.args.base_url,
            "--only", self.args.ramp_endpoint,
            "--start", "1",
            "--increment", str(self.args.ramp_increment),
            "--max", str(self.args.ramp_max),
            "--duration", str(self.args.ramp_duration),
            "--warmup", str(self.args.ramp_warmup),
            "--output-dir", str(staging),
            "--label", f"{variant.label()} ({fingerprint})",
        ]
        print(f"[exp {number:02d}] applying {variant.label()} ({fingerprint}) — ramping...")
        log_path = staging / "stdout.log"
        try:
            with log_path.open("w") as log:
                completed = subprocess.run(
                    cmd, stdout=log, stderr=subprocess.STDOUT, timeout=self.args.ramp_timeout
                )
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            return ExperimentResult(number, variant.label(), fingerprint, 0.0, 0, staging.name,
                                    "crashed", f"ramp timed out after {self.args.ramp_timeout}s",
                                    duration_seconds=time.monotonic() - started, diff=diff)
        except Exception as exc:
            return ExperimentResult(number, variant.label(), fingerprint, 0.0, 0, staging.name,
                                    "crashed", repr(exc),
                                    duration_seconds=time.monotonic() - started, diff=diff)

        run_json = staging / "run.json"
        if returncode != 0 or not run_json.exists():
            return ExperimentResult(number, variant.label(), fingerprint, 0.0, 0, staging.name,
                                    "ramp-failed", f"ramp-up.py exit {returncode}",
                                    duration_seconds=time.monotonic() - started, diff=diff)

        data = json.loads(run_json.read_text())
        result = data.get("result", {})
        peak_rps = float(result.get("peak_aggregate_rps") or 0.0)
        peak = result.get("per_endpoint_peak", {}).get(self.args.ramp_endpoint, {})

        final = self.final_dir(number, peak_rps)
        if final.exists():
            shutil.rmtree(final)
        staging.rename(final)

        return ExperimentResult(
            number=number,
            variant_label=variant.label(),
            fingerprint=fingerprint,
            peak_rps=peak_rps,
            peak_workers=result.get("peak_workers") or 0,
            folder=final.name,
            status="ok",
            avg_batch_size=peak.get("avg_batch_size"),
            p50_ms=peak.get("p50_ms"),
            p99_ms=peak.get("p99_ms"),
            stages=result.get("stages_measured", 0),
            stop_reason=result.get("stop_reason", ""),
            duration_seconds=time.monotonic() - started,
            rps_at=result.get("rps_at_reference_concurrency") or {},
            diff=diff,
        )

    # --- ratchet -------------------------------------------------------------
    def judge(self, result: ExperimentResult, variant: ec.Variant) -> str:
        """Decide whether this experiment becomes the new incumbent."""
        if result.status == "dry-run":
            return "dry-run"
        if result.status != "ok":
            self.consecutive_failures += 1
            return "skipped"
        self.consecutive_failures = 0

        if result.peak_rps > self.best_rps * (1 + MIN_IMPROVEMENT) or self.best_variant is None:
            first = self.best_variant is None
            self.best_rps = result.peak_rps
            self.best_variant = variant
            self.best_fingerprint = result.fingerprint
            self.consecutive_plateaus = 0
            return "baseline" if first else "IMPROVED"

        # During the 1-D sweep a non-improvement is normal — most knobs have a
        # bad side and only the descent phase can conclude we are stuck.
        if self.sweep_done:
            self.consecutive_plateaus += 1
        return "no-gain"

    # --- persistence ---------------------------------------------------------
    def write_results_tsv(self) -> None:
        path = self.campaign_dir / RESULTS_TSV
        fields = [
            "number", "variant", "fingerprint", "peak_rps", "peak_workers", "avg_batch_size",
            "p50_ms", "p99_ms", "rps_at_8", "rps_at_16", "rps_at_32",
            "stages", "status", "improvement_vs_best_pct",
            "duration_s", "folder", "detail",
        ]
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            running_best = 0.0
            for result, verdict in self.results:
                improvement = (
                    (result.peak_rps - running_best) / running_best * 100 if running_best else 0.0
                )
                if result.status == "ok" and result.peak_rps > running_best:
                    running_best = result.peak_rps
                writer.writerow(
                    {
                        "number": result.number,
                        "variant": result.variant_label,
                        "fingerprint": result.fingerprint,
                        "peak_rps": f"{result.peak_rps:.2f}",
                        "peak_workers": result.peak_workers,
                        "avg_batch_size": f"{result.avg_batch_size:.3f}" if result.avg_batch_size else "",
                        "p50_ms": f"{result.p50_ms:.2f}" if result.p50_ms else "",
                        "p99_ms": f"{result.p99_ms:.2f}" if result.p99_ms else "",
                        "rps_at_8": result.rps_at.get("8", ""),
                        "rps_at_16": result.rps_at.get("16", ""),
                        "rps_at_32": result.rps_at.get("32", ""),
                        "stages": result.stages,
                        "status": f"{result.status}/{verdict}",
                        "improvement_vs_best_pct": f"{improvement:+.2f}",
                        "duration_s": f"{result.duration_seconds:.1f}",
                        "folder": result.folder,
                        "detail": result.detail.replace("\t", " ").replace("\n", " "),
                    }
                )

    def write_campaign_json(self, stop_reason: str = "running") -> None:
        """Persist the campaign index. Called after every experiment, so a
        campaign still in progress can be analysed (and survives losing its
        terminal) rather than only once it finishes."""
        payload = {
            "schema_version": 1,
            "started_at": self.started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.monotonic() - self.started_monotonic, 1),
            "base_url": self.args.base_url,
            "model": self.model,
            "baseline_fingerprint": self.baseline_fp,
            "baseline_config": self.baseline,
            "ramp": {
                "max_workers": self.args.ramp_max,
                "increment": self.args.ramp_increment,
                "duration_seconds": self.args.ramp_duration,
                "warmup_seconds": self.args.ramp_warmup,
                "endpoint": self.args.ramp_endpoint,
            },
            "limits": {
                "max_experiments": self.args.max_experiments,
                "plateau_count": self.args.plateau_count,
                "max_failures": self.args.max_failures,
                "budget_seconds": self.args.budget,
                "min_improvement": MIN_IMPROVEMENT,
            },
            "experiments_run": len(self.results),
            "stop_reason": stop_reason,
            "best": {
                "peak_rps": round(self.best_rps, 2),
                "variant": self.best_variant.label() if self.best_variant else None,
                "fingerprint": self.best_fingerprint,
            },
            "experiments": [
                {
                    "number": r.number,
                    "variant": r.variant_label,
                    "fingerprint": r.fingerprint,
                    "peak_rps": round(r.peak_rps, 2),
                    "peak_workers": r.peak_workers,
                    "avg_batch_size": r.avg_batch_size,
                    "rps_at_reference_concurrency": r.rps_at,
                    "status": r.status,
                    "verdict": verdict,
                    "diff": r.diff,
                    "folder": r.folder,
                }
                for r, verdict in self.results
            ],
        }
        (self.campaign_dir / "campaign.json").write_text(json.dumps(payload, indent=2) + "\n")

    # --- main loop -----------------------------------------------------------
    def propose(self, queue):
        """Yield the next untried variant: sweep queue first, then descent."""
        baseline = self._require_baseline()
        for variant in queue:
            fp = ec.config_fingerprint(ec.apply_variant(baseline, variant))
            if fp in self.tried:
                continue
            return variant, fp
        # Sweep exhausted: coordinate descent from the incumbent.
        if self.best_variant is None:
            return None, None
        self.sweep_done = True
        while True:
            neighbours = ec.neighbour_variants(self.best_variant, set(self.tried.values()))
            fresh = []
            for variant in neighbours:
                fp = ec.config_fingerprint(ec.apply_variant(baseline, variant))
                if fp not in self.tried:
                    fresh.append((variant, fp))
            if not fresh:
                return None, None
            # Deterministic order keeps runs reproducible.
            fresh.sort(key=lambda pair: pair[1])
            return fresh[0]

    def should_stop(self) -> str:
        if len(self.results) >= self.args.max_experiments:
            return f"reached the experiment cap ({self.args.max_experiments})"
        if self.consecutive_failures >= self.args.max_failures:
            return (
                f"{self.consecutive_failures} consecutive experiments failed to run — "
                f"the server or model is probably unhealthy"
            )
        if self.consecutive_plateaus >= self.args.plateau_count:
            return (
                f"{self.consecutive_plateaus} consecutive experiments failed to beat "
                f"{self.best_rps:.1f} rps"
            )
        elapsed = time.monotonic() - self.started_monotonic
        if elapsed >= self.args.budget:
            return f"campaign budget exhausted ({elapsed / 60:.1f} min >= {self.args.budget / 60:.1f} min)"
        return ""

    def run(self) -> int:
        self.install_signal_guard()
        if not self.preflight():
            return 2

        self.campaign_dir.mkdir(parents=True, exist_ok=True)
        queue = ec.initial_queue(self._require_baseline())

        print("\n================ TRITON CONFIG CAMPAIGN ================")
        print(f"model          : {self.model}")
        print(f"campaign dir   : {self.campaign_dir}")
        print(f"ramp per test  : 1..{self.args.ramp_max} conn, "
              f"{self.args.ramp_duration:.0f}s/stage (~{self._estimate_ramp_seconds():.0f}s each)")
        print(f"limits         : {self.args.max_experiments} experiments, "
              f"{self.args.plateau_count} plateaus, {self.args.max_failures} failures, "
              f"{self.args.budget / 60:.0f} min budget")
        print("========================================================\n")

        stop_reason = ""
        try:
            while True:
                stop_reason = self.should_stop()
                if stop_reason:
                    break

                variant, fingerprint = self.propose(queue)
                if variant is None:
                    stop_reason = "search space exhausted"
                    break

                number = len(self.results) + 1
                self.tried[fingerprint] = variant
                result = self.run_experiment(number, variant)
                verdict = self.judge(result, variant)
                self.results.append((result, verdict))
                self.write_results_tsv()
                self.write_campaign_json()

                batch = f"{result.avg_batch_size:.2f}" if result.avg_batch_size else "-"
                print(
                    f"[exp {number:02d}] {result.variant_label:<28} "
                    f"{result.peak_rps:8.1f} rps @ {result.peak_workers:>3} conn | "
                    f"batch {batch:>5} | best {self.best_rps:8.1f} | {verdict}"
                )
        finally:
            self.write_results_tsv()
            self.write_campaign_json(stop_reason or "stopped")
            print(f"\n[campaign] {stop_reason or 'finished'}")
            print(f"[campaign] best: {self.best_rps:.1f} rps "
                  f"({self.best_variant.label() if self.best_variant else 'n/a'})")
            print(f"[campaign] index: {self.campaign_dir / RESULTS_TSV}")
            self.restore_baseline()

        return 0

    def _estimate_ramp_seconds(self) -> float:
        stages = len(range(1, self.args.ramp_max, self.args.ramp_increment)) + 1
        return stages * (self.args.ramp_duration + self.args.ramp_warmup + 3)


# =============================================================================
# == ENTRY POINT =============================================================
# =============================================================================


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Autonomous Triton model-configuration search campaign.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-url", default=TRITON_BASE_URL)
    parser.add_argument("--model", default=TARGET_MODEL, help="model whose config is optimised")
    parser.add_argument("--ramp-endpoint", default=RAMP_ENDPOINT_KEY,
                        help="ramp-up.py --only selector matching --model")
    parser.add_argument("--out", default=str(EXPERIMENTS_ROOT), help="campaign results root")
    parser.add_argument("--python", default=PYTHON, help="interpreter used to run ramp-up.py")
    parser.add_argument("--max-experiments", type=int, default=MAX_EXPERIMENTS)
    parser.add_argument("--plateau-count", type=int, default=PLATEAU_COUNT)
    parser.add_argument("--max-failures", type=int, default=MAX_CONSECUTIVE_FAILURES,
                        help="consecutive apply/ramp failures before giving up")
    parser.add_argument("--budget", type=float, default=CAMPAIGN_BUDGET_SECONDS, help="seconds")
    parser.add_argument("--ramp-max", type=int, default=RAMP_MAX_WORKERS)
    parser.add_argument("--ramp-increment", type=int, default=RAMP_INCREMENT)
    parser.add_argument("--ramp-duration", type=float, default=RAMP_DURATION)
    parser.add_argument("--ramp-warmup", type=float, default=RAMP_WARMUP)
    parser.add_argument("--ramp-timeout", type=float, default=RAMP_SUBPROCESS_TIMEOUT)
    parser.add_argument("--dry-run", action="store_true",
                        help="propose and record configs without touching the server")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    # A campaign is meant to run unattended with its output redirected to a
    # log. Python block-buffers stdout in that case, which would hide progress
    # until the process exits; line buffering keeps the log live.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args(argv)
    return Campaign(args).run()


if __name__ == "__main__":
    sys.exit(main())
