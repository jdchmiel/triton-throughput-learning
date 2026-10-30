"""
system_metrics.py — per-stage system telemetry for the ramp tests.

Throughput alone cannot tell you *what* limited a stage. A config that plateaus
at 620 rps is indistinguishable from one that plateaus at 620 rps because the
NIC is full, unless you also know what the CPU, VRAM, and network were doing
while the number was measured. This module samples all four on both machines at
every ramp stage so a plateau can be attributed to a resource.

What is collected, per stage, on each machine
-----------------------------------------------------------------------------
* **CPU per core** — from ``/proc/stat`` jiffies, so hyperthread siblings show
  up separately and "4 cores pegged" is visible as 4 rows near 100%.
* **Network utilisation** — ``/sys/class/net/*/statistics/{rx,tx}_bytes``
  differenced over the window, reported against that interface's **link speed**
  from ``/sys/class/net/*/speed``. Utilisation without the denominator is
  useless; 12 MB/s means nothing until you know the wire is 100 Mbps.
  These are *wire* bytes, which is what saturation is about. They will be
  smaller than the sum of the responses the client counted: aiohttp sends
  ``Accept-Encoding: gzip`` by default and Triton honours it, so a 38714 B
  JSON response costs 18705 B on the wire (0.483x — float-heavy JSON
  compresses well). Divide app bytes by ~2 to predict wire load.
* **VRAM** (server only) — ``nvidia-smi`` memory used/total plus GPU util.

Reaching the server
-----------------------------------------------------------------------------
The server is sampled over ssh with a single self-contained Python program
piped to ``python3 -``. One connection per stage, not per sample: an ssh
handshake is ~100-300 ms and sampling once a second would dominate the cost.
The remote program loops for the stage duration and returns every sample as
JSON, so the CPU percentages are computed here from raw jiffies and can be
re-derived from the saved file later.

Everything degrades to ``{"available": False, "error": ...}`` rather than
raising. Telemetry must never be able to fail a benchmark run.

Dependencies: standard library only.
"""

import glob
import json
import os
import platform
import subprocess
import sys
import time


# =============================================================================
# == THE REMOTE PROBE ========================================================
# =============================================================================
# Sent to `ssh <host> python3 - <duration> <interval> <gpu_interval>` on stdin.
# Deliberately dependency-free and old-Python-safe: it runs on whatever python3
# the server host happens to have, not on the benchmark's interpreter.
#
# It emits raw jiffies and byte counters, never percentages. Differencing is
# done locally so a bad sample can be re-examined after the fact.

REMOTE_PROBE = r'''
import glob
import json
import os
import socket
import subprocess
import sys
import time

duration = float(sys.argv[1])
interval = float(sys.argv[2])
gpu_interval = float(sys.argv[3])


def read_cpu():
    """Per-core jiffies. Index 0 is the aggregate 'cpu' summary line."""
    rows = []
    with open("/proc/stat") as handle:
        for line in handle:
            if not line.startswith("cpu"):
                break
            rows.append([int(x) for x in line.split()[1:]])
    return rows


def read_net():
    """Byte counters plus link state for every interface except loopback."""
    out = {}
    for path in sorted(glob.glob("/sys/class/net/*")):
        name = os.path.basename(path)
        if name == "lo":
            continue

        def read(rel, cast=str):
            try:
                with open(os.path.join(path, rel)) as handle:
                    return cast(handle.read().strip())
            except Exception:
                return None

        out[name] = {
            "rx": read(os.path.join("statistics", "rx_bytes"), int),
            "tx": read(os.path.join("statistics", "tx_bytes"), int),
            "link_mbps": read("speed", int),
            "state": read("operstate"),
        }
    return out


def read_gpu():
    """VRAM and GPU utilisation, or None when nvidia-smi is absent."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode != 0:
            return None
        gpus = []
        for line in proc.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "mem_total_mib": float(parts[2]),
                    "mem_used_mib": float(parts[3]),
                    "util_pct": float(parts[4]),
                }
            )
        return gpus or None
    except Exception:
        return None


samples = []
gpus = None
next_gpu = 0.0
started = time.time()
deadline = started + duration

while True:
    now = time.time()
    if now >= deadline:
        break
    if now >= next_gpu:
        gpus = read_gpu()
        next_gpu = now + gpu_interval
    sample = {"t": now, "cpu": read_cpu(), "net": read_net()}
    if gpus is not None:
        sample["gpu"] = gpus
    try:
        sample["loadavg"] = list(os.getloadavg())
    except Exception:
        sample["loadavg"] = None
    samples.append(sample)
    time.sleep(max(0.0, interval - (time.time() - now)))

print(json.dumps({
    "host": socket.gethostname(),
    "cores": max(0, len(samples[0]["cpu"]) - 1) if samples else 0,
    "duration": duration,
    "interval": interval,
    "samples": samples,
}))
'''


# =============================================================================
# == PARSING (shared by the local sampler and the remote payload) ============
# =============================================================================

# /proc/stat fields: user nice system idle iowait irq softirq steal guest...
# idle + iowait are the only "nothing happening" buckets. guest/guest_nice are
# already counted inside user on Linux, so adding them would double-count.
_IDLE_FIELDS = (3, 4)


def cpu_busy_percents(before_rows, after_rows):
    """Per-core busy percentage between two jiffies snapshots.

    ``before_rows[0]`` / ``after_rows[0]`` are the aggregate summary lines and
    are returned as element 0, so the result is ``[aggregate, core0, core1...]``
    and callers can drop the first entry when they want per-core only.
    """
    percents = []
    for before, after in zip(before_rows, after_rows):
        total_delta = 0
        idle_delta = 0
        for index, delta in enumerate(_field_deltas(before, after)):
            total_delta += delta
            if index in _IDLE_FIELDS:
                idle_delta += delta
        percents.append(
            round(100.0 * (total_delta - idle_delta) / total_delta, 2) if total_delta else 0.0
        )
    return percents


def _field_deltas(before, after):
    return [max(0, a - b) for a, b in zip(after, before)]


def net_rates(before, after, elapsed):
    """Per-interface rx/tx MB/s plus link speed, from two byte-counter maps."""
    out = {}
    if elapsed <= 0:
        return out
    for name, after_counters in (after or {}).items():
        before_counters = (before or {}).get(name)
        if not before_counters:
            continue
        rx = _counter_delta(before_counters.get("rx"), after_counters.get("rx"))
        tx = _counter_delta(before_counters.get("tx"), after_counters.get("tx"))
        link = _link_mbps(after_counters.get("link_mbps"))
        rx_MBps = rx / elapsed / 1e6
        tx_MBps = tx / elapsed / 1e6
        entry = {
            "rx_MBps": round(rx_MBps, 3),
            "tx_MBps": round(tx_MBps, 3),
            "total_MBps": round(rx_MBps + tx_MBps, 3),
            "rx_Mbps": round(rx_MBps * 8, 2),
            "tx_Mbps": round(tx_MBps * 8, 2),
            "link_mbps": link,
            "state": after_counters.get("state"),
        }
        # Ethernet is full duplex: rx and tx each get the whole link, so the
        # honest utilisation is the worse of the two directions, not their sum.
        if link and link > 0:
            entry["rx_util_pct"] = round(100.0 * entry["rx_Mbps"] / link, 1)
            entry["tx_util_pct"] = round(100.0 * entry["tx_Mbps"] / link, 1)
            entry["util_pct"] = max(entry["rx_util_pct"], entry["tx_util_pct"])
            entry["saturated"] = entry["util_pct"] >= LINK_SATURATION_PCT
        else:
            entry["util_pct"] = None
            entry["saturated"] = False
        out[name] = entry
    return out


def _counter_delta(before, after):
    if before is None or after is None:
        return 0
    return max(0, int(after) - int(before))


def _link_mbps(value):
    """sysfs reports -1 for links with no negotiable speed (bridges, veth)."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


# =============================================================================
# == LOCAL SAMPLING ==========================================================
# =============================================================================

def local_cpu_rows():
    with open("/proc/stat") as handle:
        rows = []
        for line in handle:
            if not line.startswith("cpu"):
                break
            rows.append([int(x) for x in line.split()[1:]])
    return rows


def local_net_counters():
    out = {}
    for path in sorted(glob.glob("/sys/class/net/*")):
        name = os.path.basename(path)
        if name == "lo":
            continue

        def read(rel, cast=str):
            try:
                with open(os.path.join(path, rel)) as handle:
                    return cast(handle.read().strip())
            except Exception:
                return None

        out[name] = {
            "rx": read(os.path.join("statistics", "rx_bytes"), int),
            "tx": read(os.path.join("statistics", "tx_bytes"), int),
            "link_mbps": read("speed", int),
            "state": read("operstate"),
        }
    return out


def detect_route_iface(dest_ip: str) -> str:
    """Which local interface carries traffic to ``dest_ip`` ("" if unknown).

    Needed because a machine with wifi + a USB adapter + docker bridges has
    several plausible interfaces and only one of them is the one that can be
    the bottleneck. Summing all of them would hide a saturated link inside
    idle ones.
    """
    try:
        proc = subprocess.run(
            ["ip", "route", "get", dest_ip], capture_output=True, text=True, timeout=5
        )
        parts = proc.stdout.split()
        if "dev" in parts:
            return parts[parts.index("dev") + 1]
    except Exception:
        pass
    return ""


def busiest_interface(rates: dict, preferred: str = "") -> dict:
    """The interface that matters: the routed one if it carried traffic, else
    whichever moved the most bytes. Returns {} when nothing is usable."""
    if preferred and rates.get(preferred):
        entry = dict(rates[preferred])
        entry["iface"] = preferred
        entry["chosen_by"] = "route"
        return entry
    live = {name: r for name, r in rates.items() if r.get("total_MBps", 0) > 0}
    if not live:
        return {}
    name = max(live, key=lambda k: live[k]["total_MBps"])
    entry = dict(live[name])
    entry["iface"] = name
    entry["chosen_by"] = "busiest"
    return entry


# =============================================================================
# == SAMPLING ONE STAGE ======================================================
# =============================================================================

# A link is called saturated at this fraction of its negotiated speed. Not 100%
# because TCP overheads mean a genuinely full link reports slightly under the
# nominal rate — the 100 Mbps adapter in this setup measured 103 Mbps of
# rx+tx, i.e. ~98% of the wire in the receive direction alone.
LINK_SATURATION_PCT = 90.0


class StageSampler:
    """Collects telemetry for exactly one ramp stage.

    Usage inside the stage loop::

        sampler = StageSampler(interval=1.0, ssh_host="7700x", ...)
        sampler.start()          # non-blocking, samples during the stage
        ...run load for stage_duration...
        stage_system = await sampler.finish()

    The remote probe is launched as a subprocess in a worker thread so the
    asyncio load generator is never blocked by ssh.
    """

    def __init__(
        self,
        interval: float = 1.0,
        ssh_host: str = "",
        duration: float = 10.0,
        gpu_interval: float = 2.0,
        primary_iface: str = "",
        ssh_user: str = "",
    ):
        self.interval = interval
        self.ssh_host = ssh_host
        self.duration = duration
        self.gpu_interval = gpu_interval
        self.primary_iface = primary_iface
        self.ssh_user = ssh_user
        self._samples = []
        self._stop = False
        self._remote_result = None
        self._remote_error = ""

    # --- local ---------------------------------------------------------------

    def _sample_local(self):
        self._samples.append(
            {
                "t": time.time(),
                "cpu": local_cpu_rows(),
                "net": local_net_counters(),
            }
        )

    def start(self):
        """Take the first local sample and kick off the remote probe."""
        self._sample_local()
        if self.ssh_host:
            # Imported lazily: ramp-up.py imports this module at startup and
            # asyncio.to_thread only exists on 3.9+, which is fine but keeps
            # the import list honest about what is optional.
            import asyncio

            self._remote_task = asyncio.create_task(
                asyncio.to_thread(
                    probe_remote, self.ssh_host, self.duration, self.interval, self.gpu_interval, self.ssh_user
                )
            )
        else:
            self._remote_task = None

    def tick(self):
        """Called by the stage loop while it waits, to take periodic samples."""
        if self._samples and (time.time() - self._samples[-1]["t"]) >= self.interval:
            self._sample_local()

    async def finish(self) -> dict:
        import asyncio

        self._stop = True
        self._sample_local()
        if self._remote_task is not None:
            try:
                self._remote_result, self._remote_error = await self._remote_task
            except Exception as exc:  # telemetry must never fail a run
                self._remote_result, self._remote_error = None, repr(exc)
        return self.summarise()

    # --- rollup --------------------------------------------------------------

    def summarise(self) -> dict:
        payload = {
            "available": False,
            "interval_seconds": self.interval,
            "client": summarise_samples(
                self._samples, self.primary_iface, hostname=platform.node()
            ),
            "server": None,
        }
        if self._remote_result:
            payload["server"] = summarise_samples(
                self._remote_result.get("samples", []),
                "",
                hostname=self._remote_result.get("host", self.ssh_host),
                include_gpu=True,
            )
        elif self.ssh_host:
            payload["server"] = {"available": False, "error": self._remote_error or "no samples"}

        payload["available"] = bool(payload["client"]["available"] or payload["server"])
        payload["bottleneck"] = classify(payload)
        return payload


def probe_remote(ssh_host, duration, interval, gpu_interval, ssh_user=""):
    """Run REMOTE_PROBE on the server. Returns (parsed_dict | None, error_str).

    The program travels on stdin rather than inside the ssh command line to
    avoid a second layer of shell quoting around the Python source.
    """
    target = f"{ssh_user}@{ssh_host}" if ssh_user else ssh_host
    cmd = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=5",
        target,
        "python3", "-",
        str(duration), str(interval), str(gpu_interval),
    ]
    try:
        proc = subprocess.run(
            cmd, input=REMOTE_PROBE, capture_output=True, text=True,
            timeout=duration + 30,
        )
    except subprocess.TimeoutExpired:
        return None, f"ssh probe timed out after {duration + 30}s"
    except Exception as exc:
        return None, repr(exc)

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return None, f"ssh exit {proc.returncode}: {detail[-1] if detail else 'no output'}"
    try:
        return json.loads(proc.stdout), ""
    except Exception as exc:
        return None, f"unparseable probe output ({exc})"


def summarise_samples(samples, primary_iface="", hostname="", include_gpu=False):
    """Collapse a list of raw samples into per-stage rollups.

    CPU percentages are per core, averaged over the stage, with the per-core
    maximum kept separately — mean alone hides a single pegged core behind
    fifteen idle ones.
    """
    if not samples or len(samples) < 2:
        return {"available": False, "error": "fewer than 2 samples", "hostname": hostname}

    per_core_runs = []
    net_runs = []
    for before, after in zip(samples, samples[1:]):
        elapsed = after["t"] - before["t"]
        if elapsed <= 0:
            continue
        per_core_runs.append(cpu_busy_percents(before["cpu"], after["cpu"]))
        net_runs.append((net_rates(before["net"], after["net"], elapsed), elapsed))

    if not per_core_runs:
        return {"available": False, "error": "no usable intervals", "hostname": hostname}

    core_count = len(per_core_runs[0]) - 1  # drop the aggregate row
    per_core_mean = [
        round(sum(run[1:][i] for run in per_core_runs) / len(per_core_runs), 1)
        for i in range(core_count)
    ]
    per_core_max = [
        round(max(run[1:][i] for run in per_core_runs), 1) for i in range(core_count)
    ]
    aggregate = [run[0] for run in per_core_runs]

    # Network: average the per-interval rates, weighted by interval length.
    total_seconds = sum(elapsed for _rates, elapsed in net_runs)
    merged = {}
    for rates, elapsed in net_runs:
        for name, entry in rates.items():
            slot = merged.setdefault(name, {})
            for key, value in entry.items():
                if isinstance(value, (int, float)):
                    slot[key] = slot.get(key, 0.0) + value * elapsed
    averaged = {
        name: {k: round(v / total_seconds, 3) if isinstance(v, float) else v for k, v in entry.items()}
        for name, entry in merged.items()
    }
    # link speed is a constant, not a rate — take the live value, not the mean.
    for name, entry in averaged.items():
        latest = net_runs[-1][0].get(name, {})
        entry["link_mbps"] = latest.get("link_mbps")
        entry["state"] = latest.get("state")
        link = entry.get("link_mbps")
        if link:
            entry["rx_util_pct"] = round(100.0 * entry["rx_Mbps"] / link, 1)
            entry["tx_util_pct"] = round(100.0 * entry["tx_Mbps"] / link, 1)
            entry["util_pct"] = max(entry["rx_util_pct"], entry["tx_util_pct"])
            entry["saturated"] = entry["util_pct"] >= LINK_SATURATION_PCT
        else:
            entry["util_pct"] = None
            entry["saturated"] = False

    result = {
        "available": True,
        "hostname": hostname,
        "cores": core_count,
        "samples": len(samples),
        "cpu_aggregate_pct": round(sum(aggregate) / len(aggregate), 1),
        "cpu_per_core_mean_pct": per_core_mean,
        "cpu_per_core_max_pct": per_core_max,
        # "4 cores pegging at 100%" is the observation this field exists for.
        "cores_at_95pct": sum(1 for value in per_core_max if value >= 95.0),
        # The max-based count above also fires on a single-core transient — a
        # 1% aggregate load with one sshd spike reads as "1 core pegged". A
        # core that capped throughput stayed >=95% for the *whole* stage, so
        # the mean-based count is the one safe to call a bottleneck.
        "cores_sustained_95pct": sum(1 for value in per_core_mean if value >= 95.0),
        "net": averaged,
        "net_primary": busiest_interface(averaged, primary_iface),
    }
    if samples[0].get("loadavg") is not None:
        result["loadavg_1m"] = samples[-1]["loadavg"][0]

    if include_gpu:
        gpu_runs = [s["gpu"] for s in samples if s.get("gpu")]
        if gpu_runs:
            # Multi-GPU hosts: keep per-device rollups and use device 0 for the
            # single-number summary, matching the instance_group gpus: [0].
            devices = {}
            for snapshot in gpu_runs:
                for gpu in snapshot:
                    devices.setdefault(gpu["index"], []).append(gpu)
            result["gpu"] = {
                str(index): {
                    "name": runs[0]["name"],
                    "mem_total_mib": round(sum(r["mem_total_mib"] for r in runs) / len(runs), 1),
                    "mem_used_mib_mean": round(sum(r["mem_used_mib"] for r in runs) / len(runs), 1),
                    "mem_used_mib_max": round(max(r["mem_used_mib"] for r in runs), 1),
                    "util_pct_mean": round(sum(r["util_pct"] for r in runs) / len(runs), 1),
                    "util_pct_max": round(max(r["util_pct"] for r in runs), 1),
                }
                for index, runs in sorted(devices.items())
            }
            first = result["gpu"].get("0") or next(iter(result["gpu"].values()))
            result["vram_used_mib"] = first["mem_used_mib_mean"]
            result["vram_total_mib"] = first["mem_total_mib"]
            result["gpu_util_pct"] = first["util_pct_mean"]
        else:
            result["gpu"] = None
            result["vram_error"] = "nvidia-smi returned nothing"
    return result


def classify(payload: dict) -> dict:
    """Name the resource that looks like it capped this stage.

    Deliberately conservative: it reports every resource that is close to its
    ceiling rather than picking one, because several can be tight at once and
    the point of the field is to stop a plateau being read as a model result.
    """
    signals = []
    for side in ("client", "server"):
        machine = payload.get(side)
        if not machine or not machine.get("available"):
            continue
        cores = machine.get("cores") or 0
        sustained = machine.get("cores_sustained_95pct", 0)
        if sustained:
            signals.append(
                f"{side}: {sustained}/{cores} cores pinned >=95% for the whole stage"
            )
        primary = machine.get("net_primary") or {}
        if primary.get("saturated"):
            signals.append(
                f"{side}: {primary['iface']} at {primary['util_pct']}% of a "
                f"{primary['link_mbps']} Mbps link"
            )
        if machine.get("vram_used_mib") and machine.get("vram_total_mib"):
            used_pct = 100.0 * machine["vram_used_mib"] / machine["vram_total_mib"]
            if used_pct >= 90.0:
                signals.append(f"{side}: VRAM {used_pct:.0f}% full")
    return {"saturated": bool(signals), "signals": signals}


# =============================================================================
# == CLI: sanity-check the plumbing without running a benchmark ==============
# =============================================================================

def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Sample system metrics once, for debugging.")
    parser.add_argument("--base-url", default="http://192.168.1.151:8000")
    parser.add_argument("--server-ssh", default="", help="ssh host of the Triton box; '' = local only")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args(argv)

    host = args.base_url.split("//")[-1].split(":")[0]
    iface = detect_route_iface(host)
    # stderr, so stdout stays parseable JSON for callers piping into jq.
    print(f"route to {host} -> {iface or 'unknown'}", file=sys.stderr)

    import asyncio

    async def run():
        sampler = StageSampler(
            interval=args.interval, ssh_host=args.server_ssh, duration=args.duration,
            primary_iface=iface,
        )
        sampler.start()
        deadline = time.perf_counter() + args.duration
        while time.perf_counter() < deadline:
            sampler.tick()
            await asyncio.sleep(0.1)
        return await sampler.finish()

    payload = asyncio.run(run())
    print(json.dumps(payload, indent=2))
    return 0 if payload["available"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
