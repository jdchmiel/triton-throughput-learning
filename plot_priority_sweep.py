#!/usr/bin/env python3
"""Plot the co-tenancy sweep: does search latency stay flat as indexing maxes out?

The claim being tested is a stability claim, so the figure leads with the direct
comparison and then shows the evidence that earned it:

  0. search p99 for BOTH arms on one log axis, with total offered load behind
     them -- the answer. Per-arm panels cannot show a *difference*, and the
     difference is the whole point.
  1. search latency (p50 / p95 / p99) against the SLO, per arm
  2. search delivered TPS                               -- was the load really held?
  3. indexing offered vs delivered                      -- did the load really max out?

Panels 2 and 3 are not decoration. A flat latency line is worthless if the
search load quietly dropped, and it is equally worthless if indexing never
actually threatened the server. Both have to be visibly true on the same time
axis as the latency line.

The latency axis is logarithmic. The interesting range spans a 14 ms idle floor
and a 1400 ms collapse; a linear axis either clips the collapse (so both arms
look identically "off the top") or flattens the stable region into noise.

Percentiles come from the JSONL's time buckets, whose width the generator
records. That matters: a p99 computed from ~100 samples in a 1s bucket swings
by tens of milliseconds from sampling noise alone, which would make a stable
service look unstable. The default bucket is 5s.

Usage:
  python3 plot_priority_sweep.py stella-results/priority-sweep.jsonl \
    stella-results/priority-sweep.png
"""
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

SLO_MS = float(os.environ.get("SLO_MS", 100))
SEARCH_RATE = float(os.environ.get("SEARCH_RATE", 100))


def read_log(path):
    """Return {label: {"rows": [...], "summaries": {...}}}."""
    modes = defaultdict(lambda: {"rows": [], "summaries": {}})
    for lineno, line in enumerate(open(path), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"skipping line {lineno}: {exc}", file=sys.stderr)
            continue
        label = row.get("label", "?")
        if row.get("class") in ("search", "indexing"):
            modes[label]["rows"].append(row)
        elif row.get("class") == "summary":
            # Calibration summaries share a label prefix; keep the longest run
            # per label, which is the ramp rather than the short calibration.
            prev = modes[label]["summaries"].get(row["class"])
            if prev is None or row.get("duration", 0) >= prev.get("duration", 0):
                modes[label]["summaries"]["summary"] = row
    for m in modes.values():
        m["rows"].sort(key=lambda r: (r["t"], r["class"]))
    return modes


def series(rows, cls, key):
    xs, ys = [], []
    for r in rows:
        if r["class"] != cls or r.get(key) is None:
            continue
        xs.append(r["t"])
        ys.append(r[key])
    return xs, ys


def panels(axs, rows, title, bucket):
    ax_lat, ax_tps, ax_load = axs

    sx, sp50 = series(rows, "search", "p50")
    _, sp95 = series(rows, "search", "p95")
    _, sp99 = series(rows, "search", "p99")
    _, smax = series(rows, "search", "max")
    ix, itps = series(rows, "indexing", "tps")
    _, ioff = series(rows, "indexing", "offered_rate")

    # --- panel 1: the headline -------------------------------------------
    # p99 is the SLO quantity, so it gets the heaviest line; p50 shows the
    # median did not drift either, which is what "stable" has to mean.
    ax_lat.fill_between(sx, sp50, sp99, alpha=0.15, color="#7570b3")
    ax_lat.plot(sx, sp50, color="#7570b3", lw=1.2, ls=":", label="search p50")
    if sp95:
        ax_lat.plot(sx, sp95, color="#9b8ec4", lw=1.2, label="search p95")
    ax_lat.plot(sx, sp99, color="#4b3f8f", lw=2.2, label="search p99")
    if smax:
        ax_lat.plot(sx, smax, color="#c0392b", lw=0.9, alpha=0.7,
                    marker=".", ms=3, label="search max")
    ax_lat.axhline(SLO_MS, color="#e7298a", ls="--", lw=1.4,
                   label=f"SLO {SLO_MS:.0f} ms")
    ax_lat.set_ylabel("search latency (ms)")
    ax_lat.grid(alpha=0.3, which="both")
    ax_lat.set_title(f"{title}  (buckets of {bucket:g}s)")
    ax_lat.legend(loc="upper left", fontsize=7, ncol=3, framealpha=0.9)
    # Log scale, deliberately. A linear axis has to be clipped to a few times
    # the SLO or one saturation bucket flattens the stable region to a flat
    # line -- and clipping is exactly what hides the difference between "degrades
    # to 1400ms" and "degrades to 850ms", which is the whole result. Log shows
    # the 14ms floor and the 1400ms collapse on the same axis, and the SLO line
    # still reads as a threshold.
    ax_lat.set_yscale("log")
    ax_lat.set_ylim(5, max(2000, (max(smax + sp99, default=100) * 1.6)))
    ax_lat.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_lat.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax_lat.tick_params(axis="y", which="both", labelsize=7)

    # --- panel 2: was the load actually held? ----------------------------
    _, stps = series(rows, "search", "tps")
    ax_tps.step(sx, stps, where="post", color="#1b9e77", lw=1.8,
                label="search delivered TPS")
    ax_tps.axhline(SEARCH_RATE, color="#1b9e77", ls=":", lw=1.0, alpha=0.6)
    ax_tps.set_ylabel("search TPS")
    ax_tps.set_ylim(0, SEARCH_RATE * 1.35)
    ax_tps.grid(alpha=0.3)
    ax_tps.legend(loc="lower left", fontsize=7)

    # --- panel 3: did indexing actually max out? -------------------------
    # Offered is the load the experiment applied; delivered is what the server
    # managed. The gap between them IS the saturation, and it is the part of
    # the picture that says "this is as hard as it gets".
    ax_load.step(ix, ioff, where="post", color="#d95f02", lw=2.0,
                 label="indexing offered (load applied)")
    ax_load.fill_between(ix, itps, alpha=0.20, color="#d95f02", step="post")
    ax_load.step(ix, itps, where="post", color="#a33f02", lw=1.4,
                 label="indexing delivered")
    ax_load.set_ylabel("indexing TPS")
    ax_load.set_xlabel("seconds from start of run")
    ax_load.set_ylim(bottom=0)
    ax_load.grid(alpha=0.3)
    ax_load.legend(loc="upper left", fontsize=7)


def comparison(ax, modes, names, bucket):
    """Overlay the arms: the one panel that answers the question directly.

    Per-mode panels show each run honestly; they cannot show the *difference*,
    and the difference is the deliverable. Both arms' search p99 go on one axis
    against the indexing load that caused it, so a flat line can be checked
    against the load that was supposed to bend it.

    The grey line is *total* offered load (search + indexing). Both arms bend
    at the same total, which is the finding: priority does not move the cliff,
    it changes what happens on the far side of it.
    """
    colors = ["#c0392b", "#1b6ca8", "#2e7d32"]
    total = {}
    for name in names:
        rows = modes[name]["rows"]
        by_t = defaultdict(dict)
        for r in rows:
            by_t[r["t"]][r["class"]] = r.get("offered_rate") or 0.0
        total[name] = sorted((t, d.get("search", 0.0) + d.get("indexing", 0.0))
                             for t, d in by_t.items())

    ax_load = ax.twinx()
    for color, name in zip(colors, names):
        xs, p99 = series(modes[name]["rows"], "search", "p99")
        if not xs:
            continue
        ax.step(xs, p99, where="post", color=color, lw=2.0, label=f"{name} p99")
        tx, ty = zip(*total[name])
        ax_load.plot(tx, ty, color=color, lw=1.0, ls=":", alpha=0.55)
    ax_load.set_ylabel("total offered (rps)", fontsize=7, color="#555555")
    ax_load.tick_params(axis="y", labelsize=7, colors="#555555")
    ax_load.set_ylim(0, 200)
    ax.axhline(SLO_MS, color="#e7298a", ls="--", lw=1.4, label=f"SLO {SLO_MS:.0f} ms")
    ax.set_yscale("log")
    ax.set_ylim(5, 3000)
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.tick_params(axis="y", which="both", labelsize=7)
    ax.set_ylabel("search p99 (ms)")
    ax.grid(alpha=0.3, which="both")
    ax.set_title("search p99, both arms  (buckets of %gs)" % bucket)
    ax.legend(loc="upper left", fontsize=7, ncol=3, framealpha=0.9)


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 1
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "priority-sweep.png"
    modes = read_log(src)
    # Calibration rows carry the same label with a "-cal" suffix and would
    # otherwise be plotted as extra panels.
    names = [n for n in modes if not n.endswith("-cal")]
    if not names:
        print(f"no per-second search/indexing rows in {src}", file=sys.stderr)
        return 1

    nrows = 3 * len(names) + 1
    fig, axes = plt.subplots(nrows, 1, figsize=(11, 2.6 + 8.6 * len(names)),
                             sharex=True, squeeze=False)
    buckets = [modes[n]["summaries"].get("summary", {}).get("bucket_seconds", 5.0)
               for n in names]
    comparison(axes.flat[0], modes, names, max(buckets))
    print(f"{'mode':<14} {'search TPS':>10} {'p99 med':>9} {'p99 max':>9} "
          f"{'SLO':>6} {'idx offered':>12} {'idx deliv':>10} {'pool wait':>10}")
    for i, name in enumerate(names):
        rows = modes[name]["rows"]
        summ = modes[name]["summaries"].get("summary", {})
        bucket = summ.get("bucket_seconds", 5.0)
        # squeeze=False gives (nrows, 1); flat indexing keeps this correct for
        # one mode or several without caring about the column dimension.
        base = 1 + 3 * i
        axs = [axes.flat[base], axes.flat[base + 1], axes.flat[base + 2]]
        panels(axs, rows, name, bucket)

        # Same exclusion the sweep table uses: bucket 0 straddles warmup.
        srows = [r for r in rows if r["class"] == "search" and r["t"] >= bucket]
        irows = [r for r in rows if r["class"] == "indexing"]
        if not srows:
            continue
        p99s = [r["p99"] for r in srows]
        # Median of the bucket p99s is the stability statistic; the max is the
        # violation. A service can be stable and still have one bad bucket,
        # and those two facts need to be reported separately.
        med = sorted(p99s)[len(p99s) // 2]
        verdict = "ok" if max(p99s) <= SLO_MS else "BREAK"
        print(f"{name:<14} "
              f"{summ.get('search_delivered', 0):>10} "
              f"{med:>9.1f} {max(p99s):>9.1f} {verdict:>6} "
              f"{max((r['offered_rate'] for r in irows), default=0):>12.0f} "
              f"{summ.get('indexing_delivered', 0):>10} "
              f"{max(summ.get('search_pool_wait_max', 0), summ.get('indexing_pool_wait_max', 0)):>9.1f}ms")

    fig.suptitle("stella400m co-tenancy: search latency under an indexing ramp",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
