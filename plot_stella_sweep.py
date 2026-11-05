#!/usr/bin/env python3
"""Plot the stella400m sweep from the table in PROGRESS_LOG.md.

Three panels, one per question the campaign asks:
  throughput  -- did we get more requests out of the box
  GPU util    -- was the GPU actually the thing holding us back
  VRAM        -- what did each instance_group copy cost

The data lives in PROGRESS_LOG.md rather than a CSV because that file is the
durable record of the campaign; the raw sweep log on the host is scratch and
gets cleaned. Parsing the table means this plot can be regenerated from a
clean clone.

Reads the table under the "2026-11-05" heading whose header row contains
"req/exec". Rows are keyed on (config, conc) so re-running the sweep and
re-editing the table is all it takes to refresh the figure.

    ./plot_stella_sweep.py [--log PROGRESS_LOG.md] [--out stella-sweep.png]
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The three rows measured at instance_group count 1 all landed within 1% of
# each other (110/117/116 rps) while GPU utilisation stayed pinned near 35%.
# That flat line is NOT the wifi link: sweep.sh runs the load generator
# *inside* the container with `docker exec`, so the laptop's NIC is nowhere in
# the measurement path. It is the single python-backend instance serving
# requests one at a time -- latency scales exactly linearly with concurrency
# (8.98 ms at conc=1, 136 ms at 16, 553 ms at 64) while throughput does not
# move at all, which is the signature of one serial service point of ~8.6 ms.
# The GPU has headroom, which is why ic2/ic4 scale and why the gains are
# believable (sm moved 35% -> 82% alongside them).
CLIENT_CEILING_RPS = 116.0
CEILING_LABEL = "ic1 serial-instance ceiling (~116 rps)\nnot the network: load runs in-container"

VRAM_TOTAL_MB = 10240  # RTX 3080 10 GB


def read_log(path):
    p = Path(path)
    if not p.exists():
        sys.exit(f"no log at {p} (run from the repo root, or pass --log)")
    return p.read_text()


def parse_table(log_text):
    """Return list of dicts from the stella sweep markdown table."""
    lines = log_text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("|") and "req/exec" in line:
            start = i
            break
    if start is None:
        sys.exit("no sweep table found in log (looking for a header with 'req/exec')")

    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 8:
            break
        rows.append(
            {
                "config": cells[0].replace("*", ""),
                "conc": int(cells[1]),
                "rps": float(cells[2].replace("*", "")),
                "p50_ms": float(cells[3].replace(" ms", "")),
                "req_per_exec": float(cells[5]),
                "sm_avg": float(re.search(r"(\d+)%", cells[6]).group(1)),
                "sm_max": float(re.search(r"\((\d+)%\)", cells[6]).group(1)),
                "fb_mb": float(cells[7].replace("MB", "").strip()),
            }
        )
    if not rows:
        sys.exit("sweep table found but no rows parsed")
    return rows


def row_label(row):
    """Short x-axis label: 'ic1 c16' style, derived from the config text."""
    m = re.search(r"ic(\d+)", row["config"])
    ic = f"ic{m.group(1)}" if m else "ic1"
    return f"{ic}\nconc={row['conc']}"


def annotate_axes(ax, title, ylabel):
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="PROGRESS_LOG.md")
    ap.add_argument("--out", default="stella-sweep.png")
    args = ap.parse_args()

    rows = parse_table(read_log(args.log))
    labels = [row_label(r) for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    fig.suptitle(
        "stella400m on Triton (RTX 3080, fp16, 20s windows)\n"
        "instance_group.count is the only lever that moves throughput",
        fontsize=13,
        fontweight="bold",
    )

    # --- throughput -------------------------------------------------------
    ax = axes[0]
    colors = ["#c0392b" if "ic1" in r["config"] else "#2980b9" for r in rows]
    ax.bar(labels, [r["rps"] for r in rows], color=colors, alpha=0.85)
    ax.axhline(CLIENT_CEILING_RPS, color="#7f8c8d", ls="--", lw=1.2)
    # Empty space above the short ic1 bars, clear of the tall ic2/ic4 bars.
    ax.text(
        0.02,
        0.62,
        CEILING_LABEL,
        transform=ax.transAxes,
        fontsize=8,
        color="#7f8c8d",
        va="top",
        ha="left",
    )
    for i, r in enumerate(rows):
        ax.text(i, r["rps"] + 4, f'{r["rps"]:.0f}', ha="center", fontsize=8)
    annotate_axes(ax, "Throughput", "requests / sec")
    ax.set_ylim(0, max(r["rps"] for r in rows) * 1.28)

    # --- GPU utilisation --------------------------------------------------
    ax = axes[1]
    x = range(len(rows))
    ax.bar(
        [i - 0.2 for i in x],
        [r["sm_avg"] for r in rows],
        width=0.4,
        color="#16a085",
        alpha=0.9,
        label="sm avg",
    )
    ax.bar(
        [i + 0.2 for i in x],
        [r["sm_max"] for r in rows],
        width=0.4,
        color="#16a085",
        alpha=0.45,
        label="sm max",
    )
    ax.axhline(
        35,
        color="#c0392b",
        ls=":",
        lw=1.2,
        label="ic1 ceiling (~35%) regardless of load",
    )
    for i, r in enumerate(rows):
        ax.text(i - 0.2, r["sm_avg"] + 1.5, f'{r["sm_avg"]:.0f}', ha="center", fontsize=7)
    annotate_axes(ax, "GPU utilisation (nvidia-smi dmon -s um, sm column)", "% of SM busy")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=7, loc="upper left", ncol=3)

    # --- VRAM -------------------------------------------------------------
    ax = axes[2]
    ax.bar(labels, [r["fb_mb"] for r in rows], color="#8e44ad", alpha=0.85)
    ax.axhline(VRAM_TOTAL_MB, color="#7f8c8d", ls="--", lw=1.2)
    # Empty band between the tallest bars (4795 MB) and the total (10240 MB).
    ax.text(
        0.02,
        0.72,
        "10240 MB total -- ic4 uses less than half, so VRAM is not the limit",
        transform=ax.transAxes,
        fontsize=8,
        color="#7f8c8d",
        va="top",
        ha="left",
    )
    for i, r in enumerate(rows):
        ax.text(i, r["fb_mb"] + 120, f'{r["fb_mb"]:.0f}', ha="center", fontsize=8)
    annotate_axes(ax, "VRAM footprint (framebuffer)", "MB")
    ax.set_ylim(0, VRAM_TOTAL_MB * 1.18)
    ax.set_xlabel("config (ic = instance_group.count)")

    fig.text(
        0.5,
        0.005,
        "The flat ~116 rps across all ic1 rows is the single python-backend instance serving "
        "requests serially, not the network -- loadgen runs inside the container.\nic2/ic4 "
        "gains are real because sm utilisation moved 35% -> 82% with them. Absolute rps is "
        "still unverified against a client-side ramp-up run, which IS network-bound.",
        ha="center",
        fontsize=8,
        style="italic",
        color="#555555",
    )

    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out} from {len(rows)} sweep rows")


if __name__ == "__main__":
    main()
