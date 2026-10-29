"""
analyze.py — summarise a finished campaign.

Reads a campaign directory (the default is the most recent one under
``experiments/``) and prints the leaderboard, the improvement trail, and the
per-knob marginal effect. Also renders ``progress.png``: peak throughput over
experiment order with the ratchet marked, plus realised batch size.

    python analyze.py                       # newest campaign
    python analyze.py experiments/20260101-120000
    python analyze.py --campaign experiments/... --open
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXPERIMENTS_ROOT = Path("experiments")
PROGRESS_PNG = "progress.png"
PLOT_DPI = 130


def find_latest_campaign(root: Path) -> Path:
    if not root.exists():
        raise SystemExit(f"no experiments directory at {root} — run campaign.py first")
    candidates = sorted(p for p in root.iterdir() if (p / "campaign.json").exists())
    if not candidates:
        raise SystemExit(f"no campaign.json under {root} — run campaign.py first")
    return candidates[-1]


def load_rows(campaign_dir: Path) -> list:
    """Read results.tsv into dicts with numbers coerced."""
    path = campaign_dir / "results.tsv"
    if not path.exists():
        raise SystemExit(f"{path} not found")
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            for key in ("peak_rps", "avg_batch_size", "p50_ms", "p99_ms",
                        "improvement_vs_best_pct", "duration_s",
                        "rps_at_8", "rps_at_16", "rps_at_32"):
                row[key] = float(row[key]) if row[key] not in ("", None) else None
            for key in ("number", "peak_workers", "stages"):
                row[key] = int(row[key]) if row[key] not in ("", None) else None
            rows.append(row)
    return rows


def verdict_of(row: dict) -> str:
    """The keep/discard verdict, stored as the suffix of results.tsv status."""
    status = row.get("status") or ""
    return status.split("/", 1)[1] if "/" in status else status


def accepted(row: dict) -> bool:
    """Did the campaign actually adopt this config as the new incumbent?

    Must be read from the verdict, not inferred from the peak: a config can
    beat the previous best and still be rejected for missing the 2% ratchet.
    """
    return verdict_of(row) in ("IMPROVED", "baseline")


def knob_marginals(rows: list) -> dict:
    """Best peak seen for each value of each single-knob experiment.

    Only 1-D sweep rows (exactly one knob) are considered, so the numbers read
    as "what did changing just this knob get us".
    """
    best = {k: {} for k in ("mbs", "pbs", "qd", "ic")}
    for row in rows:
        variant = row["variant"]
        if variant in ("baseline", "") or row["peak_rps"] is None:
            continue
        parts = variant.split("_")
        if len(parts) != 1:
            continue  # composite (descent) row, not a single-knob sweep
        part = parts[0]
        if part.startswith("mbs"):
            key, value = "mbs", int(part[3:])
        elif part.startswith("pbs"):
            key, value = "pbs", part.split(":")[1]
        elif part.startswith("qd"):
            key, value = "qd", int(part[2:].replace("us", ""))
        elif part.startswith("ic"):
            key, value = "ic", int(part[2:])
        else:
            continue
        current = best[key].get(value)
        if current is None or row["peak_rps"] > current["peak_rps"]:
            best[key][value] = row
    return best


def print_report(campaign_dir: Path, rows: list, meta: dict) -> None:
    baseline_rps = None
    for row in rows:
        if row["variant"] == "baseline" and row["peak_rps"]:
            baseline_rps = row["peak_rps"]
            break

    ok_rows = [r for r in rows if r["peak_rps"]]
    # Two different "best"s, and conflating them would be a lie: the config the
    # campaign actually adopted (it must clear the 2% ratchet) versus the
    # single highest number ever measured, which may have been rejected.
    adopted = [r for r in ok_rows if accepted(r)]
    winner_row = adopted[-1] if adopted else None
    peak_row = max(ok_rows, key=lambda r: r["peak_rps"]) if ok_rows else None

    print("=" * 74)
    print("TRITON CONFIG CAMPAIGN REPORT")
    print("=" * 74)
    print(f"campaign      : {campaign_dir}")
    print(f"model         : {meta.get('model')}")
    print(f"started       : {meta.get('started_at')}")
    print(f"elapsed       : {(meta.get('elapsed_seconds') or 0) / 60:.1f} min")
    print(f"experiments   : {meta.get('experiments_run', len(rows))}")
    stop_reason = meta.get("stop_reason", "running")
    print(f"stop reason   : {stop_reason}")
    if stop_reason == "running":
        print("                (campaign still in progress — numbers are partial)")
    print()

    if baseline_rps and winner_row:
        speedup = winner_row["peak_rps"] / baseline_rps
        print(f"baseline peak : {baseline_rps:.1f} rps")
        print(f"adopted best  : {winner_row['peak_rps']:.1f} rps   ({winner_row['variant']})")
        print(f"speedup       : {speedup:.2f}x")
        if winner_row["avg_batch_size"]:
            print(f"realised batch: {winner_row['avg_batch_size']:.2f} at the winning config")
        if peak_row and peak_row is not winner_row and peak_row["peak_rps"] > winner_row["peak_rps"]:
            threshold = (meta.get("limits") or {}).get("min_improvement", 0.02)
            print(f"              note: {peak_row['peak_rps']:.1f} rps was measured "
                  f"({peak_row['variant']}) but rejected — it did not clear the "
                  f"{threshold * 100:.0f}% ratchet")
        print()

    print("--- improvement trail (every accepted config) " + "-" * 28)
    for row in rows:
        if not row["peak_rps"] or not accepted(row):
            continue
        gain = f"{row['improvement_vs_best_pct']:+.1f}%" if row["improvement_vs_best_pct"] else "  base"
        print(f"  #{row['number']:>2}  {row['variant']:<38} {row['peak_rps']:8.1f} rps  {gain}")
    print()

    print("--- single-knob marginals " + "-" * 45)
    for knob, label in (("mbs", "max_batch_size"), ("pbs", "preferred_batch_size"),
                        ("qd", "queue_delay_us"), ("ic", "instance_count")):
        marginals = knob_marginals(rows)[knob]
        if not marginals:
            continue
        top = max(marginals.values(), key=lambda r: r["peak_rps"])
        line = "  ".join(f"{v}={r['peak_rps']:.0f}" for v, r in sorted(marginals.items(), key=lambda kv: str(kv[0])))
        print(f"  {label:<22} best={top['peak_rps']:7.1f} | {line}")
    print()

    print("--- leaderboard (top 10) " + "-" * 44)
    header = f"  {'peak rps':>9}  {'batch':>5}  {'@conn':>5}  {'@8':>7}  {'@16':>7}  {'@32':>7}  variant"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in sorted(ok_rows, key=lambda r: -r["peak_rps"])[:10]:
        batch = f"{row['avg_batch_size']:.2f}" if row["avg_batch_size"] else "  - "
        # A blank means the ramp plateau-stopped before it ever reached that
        # concurrency — it is not a measurement of zero, so do not print 0.
        ref = "".join(
            f"  {(f'{row[r]:.0f}' if row.get(r) else '-'):>7}" for r in (f"rps_at_{lvl}" for lvl in (8, 16, 32))
        )
        print(f"  {row['peak_rps']:9.1f}  {batch:>5}  {row['peak_workers']:>5}{ref}  {row['variant']}")
    print("  (@8/@16/@32 = aggregate RPS at that fixed concurrency: the operational tiebreaker)")
    print()

    failures = [r for r in rows if "/" in (r["status"] or "") and not r["status"].startswith("ok/")]
    if failures:
        print(f"--- experiments that did not produce a number ({len(failures)}) " + "-" * 18)
        for row in failures[:10]:
            print(f"  #{row['number']:>2}  {row['variant']:<30} {row['status']:<22} {row['detail'][:60]}")
        print()


def plot_progress(campaign_dir: Path, rows: list, meta: dict) -> Path:
    ok_rows = [r for r in rows if r["peak_rps"]]
    if not ok_rows:
        print("No measured experiments to plot.")
        return None

    numbers = [r["number"] for r in ok_rows]
    rps = [r["peak_rps"] for r in ok_rows]

    # The ratchet follows the campaign's own verdicts, not a raw maximum, so
    # the chart shows the incumbent the driver actually carried forward.
    running_best = []
    improved = []
    best = 0.0
    for row in ok_rows:
        if accepted(row):
            best = row["peak_rps"]
            improved.append(True)
        else:
            improved.append(False)
        running_best.append(best)

    baseline_rps = next((r["peak_rps"] for r in ok_rows if r["variant"] == "baseline"), None)

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    axes[0].plot(numbers, rps, marker="o", linewidth=1.4, alpha=0.55, color="tab:blue", label="experiment peak")
    axes[0].step(numbers, running_best, where="post", color="black", linewidth=2, label="best so far (ratchet)")
    if baseline_rps:
        axes[0].axhline(baseline_rps, color="tab:red", linestyle="--", linewidth=1.2,
                        label=f"baseline {baseline_rps:.0f} rps")
        axes[0].axhline(baseline_rps * 10, color="green", linestyle=":", linewidth=1.4,
                        label=f"10x target {baseline_rps * 10:.0f} rps")
    for row, was_best in zip(ok_rows, improved):
        if was_best:
            axes[0].annotate(row["variant"], (row["number"], row["peak_rps"]),
                             textcoords="offset points", xytext=(0, 8), fontsize=6, rotation=45, ha="left")
    axes[0].set_ylabel("Peak aggregate RPS")
    axes[0].set_title(f"Triton config campaign — {meta.get('model')} ({meta.get('experiments_run')} experiments)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    batches = [(r["avg_batch_size"], r["number"]) for r in ok_rows if r["avg_batch_size"]]
    if batches:
        batches.sort(key=lambda pair: pair[1])
        axes[1].plot([p[1] for p in batches], [p[0] for p in batches], marker="s", color="tab:purple", linewidth=1.4)
        axes[1].set_ylabel("Realised avg batch size")
        axes[1].set_title("Server-side batching realised by each config")
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "no server-side batch stats recorded", ha="center", va="center")
    axes[1].set_xlabel("Experiment number")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out = campaign_dir / PROGRESS_PNG
    fig.savefig(out, dpi=PLOT_DPI)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Summarise a Triton config campaign.")
    parser.add_argument("campaign", nargs="?", default=None, help="campaign directory (default: newest)")
    parser.add_argument("--root", default=str(EXPERIMENTS_ROOT), help="experiments root to search")
    args = parser.parse_args(argv)

    campaign_dir = Path(args.campaign) if args.campaign else find_latest_campaign(Path(args.root))
    meta_path = campaign_dir / "campaign.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    rows = load_rows(campaign_dir)
    print_report(campaign_dir, rows, meta)

    png = plot_progress(campaign_dir, rows, meta)
    if png:
        print(f"chart: {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
