"""
experiment_config.py — Triton config manipulation library for the campaign.

This module knows how to talk to a Triton server's model-repository API and how
to build candidate model configurations to evaluate. It is a pure library: no
orchestration, no benchmarking. ``campaign.py`` drives it; ``ramp-up.py`` does
the actual load generation.

Why the load API and not editing files on disk
----------------------------------------------
Triton can change a running model's configuration in memory via

    POST /v2/repository/models/{name}/load
        body: {"parameters": {"config": "<full config as a JSON STRING>"}}

Three facts that shape the code below (all verified against Triton 2.69.0):

* ``config`` is a *string*, not a nested object, and it is a *full replacement*,
  not a merge. So we always GET the live config, deep-copy it, mutate the few
  fields under test, and send the whole thing back.
* Runtime load/unload only works when the server runs with
  ``--model-control-mode=explicit``. In the default ``none``/poll mode every
  load call fails with "explicit model load / unload is not allowed if polling
  is enabled". ``preflight()`` detects that up front.
* Unload is *asynchronous*: the call returns before the model is gone. We must
  poll ``POST /v2/repository/index`` until the model reports
  ``state == "UNAVAILABLE"`` with ``reason == "unloaded"`` before reloading, or
  the reload races the teardown.

Dependencies: standard library only (urllib), so the driver stays dependency
light and the benchmarking deps stay isolated in ramp-up.py.
"""

import copy
import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional


# =============================================================================
# == SEARCH SPACE ============================================================
# =============================================================================
# Every dimension below is a dynamic-batching / scheduling knob that Triton can
# change at runtime through the load API. These are the levers the campaign
# pulls; ramp-up.py measures the effect.

MAX_BATCH_SIZES = [1, 8, 16, 32, 64, 128]

# "auto" means: omit preferred_batch_size entirely and let Triton batch freely.
# The explicit lists are curated subsets of the current production preference.
PREFERRED_BATCH_SIZE_SETS = {
    "auto": None,
    "tiny": [1, 2, 4, 8],
    "small": [2, 4, 8, 16],
    "mid": [8, 16, 32],
    "wide": [4, 8, 16, 32, 64, 128],
    "current": [2, 4, 8, 16, 24, 32, 48, 64, 96, 128],
}

QUEUE_DELAY_MICROSECONDS = [0, 500, 2000, 5000, 10000, 25000]

INSTANCE_COUNTS = [1, 2, 3, 4]


@dataclass(frozen=True)
class Variant:
    """One candidate configuration expressed as the knobs under test.

    A value of ``None`` for a field means "leave the baseline value alone",
    except ``preferred`` which uses the key into PREFERRED_BATCH_SIZE_SETS.
    """

    max_batch_size: Optional[int] = None
    preferred: Optional[str] = None   # key into PREFERRED_BATCH_SIZE_SETS
    queue_delay_us: Optional[int] = None
    instance_count: Optional[int] = None

    def label(self) -> str:
        parts = []
        if self.max_batch_size is not None:
            parts.append(f"mbs{self.max_batch_size}")
        if self.preferred is not None:
            parts.append(f"pbs:{self.preferred}")
        if self.queue_delay_us is not None:
            parts.append(f"qd{self.queue_delay_us}us")
        if self.instance_count is not None:
            parts.append(f"ic{self.instance_count}")
        return "_".join(parts) if parts else "baseline"


def apply_variant(baseline: dict, variant: Variant) -> dict:
    """Return a deep copy of ``baseline`` with the variant's knobs applied.

    Never mutates the baseline: the caller keeps it pristine as the restore
    target. Only the fields the variant sets are touched.
    """
    config = copy.deepcopy(baseline)

    if variant.max_batch_size is not None:
        config["max_batch_size"] = variant.max_batch_size

    if variant.preferred is not None:
        sizes = PREFERRED_BATCH_SIZE_SETS[variant.preferred]
        db = config.setdefault("dynamic_batching", {})
        if sizes is None:
            db.pop("preferred_batch_size", None)
        else:
            db["preferred_batch_size"] = list(sizes)

    if variant.queue_delay_us is not None:
        db = config.setdefault("dynamic_batching", {})
        db["max_queue_delay_microseconds"] = variant.queue_delay_us

    if variant.instance_count is not None:
        groups = config.get("instance_group") or []
        for group in groups:
            group["count"] = variant.instance_count

    normalize_config(config)
    return config


def normalize_config(config: dict) -> int:
    """Enforce the invariants Triton validates a config against, in place.

    Triton refuses to load a model whose ``preferred_batch_size`` contains a
    value above ``max_batch_size`` ("dynamic batching preferred size must be <=
    max batch size"), and the baseline preference list runs all the way up to
    the baseline ``max_batch_size``. So lowering ``max_batch_size`` on its own
    produces a config the server will reject outright — every ``mbs``-only
    experiment would fail. Truncating the preference list to what is still
    legal is the only meaningful interpretation of such a variant, and it is
    done here so the caller never sends an unloadable config.

    Returns the number of preferred sizes that had to be dropped.
    """
    db = config.get("dynamic_batching")
    if not isinstance(db, dict):
        return 0
    preferred = db.get("preferred_batch_size")
    if not isinstance(preferred, list):
        return 0

    max_batch = config.get("max_batch_size", 0) or 0
    kept = [size for size in preferred if 0 < size <= max_batch]
    dropped = len(preferred) - len(kept)
    if kept:
        db["preferred_batch_size"] = sorted(set(kept))
    else:
        # An empty list is not a valid "no preference"; omitting the field is.
        db.pop("preferred_batch_size", None)
    return dropped


def validate_config(config: dict) -> list:
    """Return a list of reasons Triton would refuse this config, empty if OK.

    Checked locally so a bad candidate is recorded as invalid immediately
    instead of costing an unload/load round trip whose HTTP error hides the
    real reason (the server returns a generic "failed to poll from model
    repository" and logs the actual complaint only to its own stderr).
    """
    problems = []
    max_batch = config.get("max_batch_size", 0) or 0
    if max_batch < 1:
        problems.append(f"max_batch_size must be >= 1, got {max_batch}")

    db = config.get("dynamic_batching") or {}
    for size in db.get("preferred_batch_size") or []:
        if size < 1:
            problems.append(f"preferred_batch_size must be >= 1, got {size}")
        elif size > max_batch:
            problems.append(
                f"preferred_batch_size {size} exceeds max_batch_size {max_batch}"
            )

    delay = db.get("max_queue_delay_microseconds")
    if delay is not None and delay < 0:
        problems.append(f"max_queue_delay_microseconds must be >= 0, got {delay}")

    for index, group in enumerate(config.get("instance_group") or []):
        count = group.get("count", 1)
        if count < 1:
            problems.append(f"instance_group[{index}].count must be >= 1, got {count}")

    if not config.get("input") or not config.get("output"):
        problems.append("config must declare at least one input and one output")
    if not config.get("name"):
        problems.append("config must carry the model name")
    return problems


def config_fingerprint(config: dict) -> str:
    """Stable hash of the fields that actually affect scheduling.

    Used to dedupe experiments: two variants that produce the same effective
    batching/scheduling behaviour should not be benchmarked twice.
    """
    db = config.get("dynamic_batching") or {}
    relevant = {
        "max_batch_size": config.get("max_batch_size"),
        "preferred_batch_size": db.get("preferred_batch_size"),
        "max_queue_delay_microseconds": db.get("max_queue_delay_microseconds"),
        "instance_group": [
            {
                "kind": g.get("kind"),
                "count": g.get("count"),
                "priority": (g.get("rate_limiter") or {}).get("priority"),
            }
            for g in (config.get("instance_group") or [])
        ],
    }
    blob = json.dumps(relevant, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def config_diff(baseline: dict, candidate: dict) -> dict:
    """Field-level diff of the scheduling-relevant parts of two configs."""
    diff = {}
    for key in ("max_batch_size",):
        if baseline.get(key) != candidate.get(key):
            diff[key] = {"from": baseline.get(key), "to": candidate.get(key)}

    base_db = baseline.get("dynamic_batching") or {}
    cand_db = candidate.get("dynamic_batching") or {}
    for key in ("preferred_batch_size", "max_queue_delay_microseconds"):
        if base_db.get(key) != cand_db.get(key):
            diff[f"dynamic_batching.{key}"] = {"from": base_db.get(key), "to": cand_db.get(key)}

    base_ic = [(g.get("count"), g.get("kind")) for g in (baseline.get("instance_group") or [])]
    cand_ic = [(g.get("count"), g.get("kind")) for g in (candidate.get("instance_group") or [])]
    if base_ic != cand_ic:
        diff["instance_group"] = {"from": base_ic, "to": cand_ic}
    return diff


# =============================================================================
# == TRITON CLIENT ===========================================================
# =============================================================================


class TritonError(RuntimeError):
    """A Triton admin API call failed; message carries the server's error."""


@dataclass
class TritonClient:
    """Synchronous client for the Triton model-repository / stats endpoints.

    Synchronous on purpose: the campaign driver spends most of its time waiting
    on the ramp-up subprocess, so an async client here buys nothing and only
    complicates Ctrl-C handling and the baseline-restore guard.
    """

    base_url: str
    timeout: float = 15.0
    poll_interval: float = 0.5
    unload_timeout: float = 350.0
    load_timeout: float = 120.0

    # --- low-level HTTP ------------------------------------------------------
    def _request(self, method: str, path: str, body=None):
        url = f"{self.base_url}{path}"
        data = None
        headers = {}
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
                return response.status, (json.loads(raw) if raw else {})
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                parsed = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                parsed = {"error": raw.decode(errors="replace")}
            return exc.code, parsed
        except urllib.error.URLError as exc:
            raise TritonError(f"cannot reach {url}: {exc.reason}") from exc

    # --- health / metadata ---------------------------------------------------
    def server_metadata(self) -> dict:
        status, body = self._request("GET", "/v2")
        return body

    def health_ready(self) -> bool:
        status, _ = self._request("GET", "/v2/health/ready")
        return status == 200

    def control_mode_probe(self) -> dict:
        """Detect whether runtime load/unload is permitted.

        Targets a deliberately nonexistent model so the probe cannot disturb the
        live models. The error text distinguishes the two failure modes:
        polling mode (needs --model-control-mode=explicit) vs a plain
        "model not found" (explicit mode is active and working).
        """
        status, body = self._request("POST", "/v2/repository/models/__triton_probe__/load", body={})
        message = (body or {}).get("error", "")
        polling = "polling is enabled" in message or "not allowed" in message
        return {
            "status": status,
            "error": message,
            "load_api_available": (not polling) and status != 0,
        }

    # --- config --------------------------------------------------------------
    def get_config(self, model: str) -> dict:
        status, body = self._request("GET", f"/v2/models/{model}/config")
        if status != 200:
            raise TritonError(f"get_config {model}: HTTP {status} {body.get('error', '')}")
        return body

    def repository_index(self, model: Optional[str] = None) -> list:
        body: dict = {"ready": False}
        if model:
            body["name"] = model
        status, entries = self._request("POST", "/v2/repository/index", body=body)
        if status != 200:
            raise TritonError(f"repository_index: HTTP {status} {entries.get('error', '')}")
        return entries if isinstance(entries, list) else []

    def model_state(self, model: str):
        for entry in self.repository_index():
            if entry.get("name") == model:
                return entry.get("state"), entry.get("reason", "")
        return None, ""

    # --- load / unload -------------------------------------------------------
    def load(self, model: str, config: Optional[dict] = None) -> None:
        """Load (or reconfigure) a model. ``config`` is a FULL replacement.

        Triton requires the config to be sent as a JSON *string* inside
        ``parameters.config``. Omitting it performs a bare load from disk.
        """
        body = None
        if config is not None:
            body = {"parameters": {"config": json.dumps(config)}}
        status, resp = self._request("POST", f"/v2/repository/models/{model}/load", body=body)
        if status not in (200, 201):
            raise TritonError(f"load {model}: HTTP {status} {resp.get('error', '')}")

    def unload(self, model: str) -> None:
        status, resp = self._request("POST", f"/v2/repository/models/{model}/unload")
        if status not in (200, 201, 204):
            raise TritonError(f"unload {model}: HTTP {status} {resp.get('error', '')}")

    def wait_until_unloaded(self, model: str, timeout: Optional[float] = None) -> None:
        """Poll until the model is fully gone (async unload finished).

        Triton's own Sagemaker server polls 500ms up to 350s; we match that.
        A model that never appears in the index is also considered unloaded.
        """
        timeout = self.unload_timeout if timeout is None else timeout
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state, reason = self.model_state(model)
            if state is None:
                return
            if state == "UNAVAILABLE" and reason == "unloaded":
                return
            time.sleep(self.poll_interval)
        raise TritonError(f"{model} did not reach UNAVAILABLE/unloaded within {timeout}s")

    def wait_until_ready(self, model: str, timeout: Optional[float] = None) -> None:
        timeout = self.load_timeout if timeout is None else timeout
        deadline = time.monotonic() + timeout
        last_reason = ""
        while time.monotonic() < deadline:
            state, reason = self.model_state(model)
            if state == "READY":
                return
            last_reason = reason
            time.sleep(self.poll_interval)
        raise TritonError(f"{model} did not reach READY within {timeout}s (last reason: {last_reason})")

    # --- stats ---------------------------------------------------------------
    def stats(self, model: str) -> dict:
        status, body = self._request("GET", f"/v2/models/{model}/stats")
        if status != 200:
            return {}
        entries = body.get("model_stats")
        if isinstance(entries, list):
            for entry in entries:
                if entry.get("name") == model:
                    return entry
            return entries[0] if entries else {}
        return body

    # --- high-level apply ----------------------------------------------------
    def apply_config(self, model: str, config: dict, verify: bool = True) -> dict:
        """Fully replace a model's running config and confirm it took effect.

        Unload-then-load (rather than a bare load) is used deliberately: a bare
        load on a READY model can take Triton's in-place UpdateModelConfig path,
        which does not reliably re-create instances when instance_group.count
        changes. A clean reload is slower but correct for every dimension.
        """        # Fail before touching the server: Triton reports every rejection in the
        # config override as the same generic "failed to poll from model
        # repository" over HTTP and logs the real reason only to its own stderr,
        # so an invalid candidate must be caught here to be diagnosable.
        problems = validate_config(config)
        if problems:
            raise TritonError("refusing to send an invalid config: " + "; ".join(problems))
        self.unload(model)
        self.wait_until_unloaded(model)
        self.load(model, config)
        self.wait_until_ready(model)

        if not verify:
            return {}
        applied = self.get_config(model)
        mismatches = {}
        for key in ("max_batch_size",):
            if applied.get(key) != config.get(key):
                mismatches[key] = {"sent": config.get(key), "applied": applied.get(key)}
        sent_db = config.get("dynamic_batching") or {}
        got_db = applied.get("dynamic_batching") or {}
        for key in ("preferred_batch_size", "max_queue_delay_microseconds"):
            if key in sent_db and got_db.get(key) != sent_db.get(key):
                mismatches[f"dynamic_batching.{key}"] = {
                    "sent": sent_db.get(key),
                    "applied": got_db.get(key),
                }
        sent_counts = [g.get("count") for g in (config.get("instance_group") or [])]
        got_counts = [g.get("count") for g in (applied.get("instance_group") or [])]
        if sent_counts and sent_counts != got_counts:
            mismatches["instance_group.count"] = {"sent": sent_counts, "applied": got_counts}
        if mismatches:
            raise TritonError(f"applied config does not match sent config: {mismatches}")
        return applied

    def restore(self, model: str, baseline: dict) -> None:
        """Best-effort restore of the pristine baseline config."""
        try:
            self.apply_config(model, baseline)
        except TritonError:
            # Last resort: a bare load re-reads the on-disk config.
            try:
                self.unload(model)
                self.wait_until_unloaded(model)
                self.load(model)
                self.wait_until_ready(model)
            except TritonError:
                pass


# =============================================================================
# == SEARCH STRATEGY =========================================================
# =============================================================================
# Mirrors the autoresearch driver: propose one variant, measure it, keep the
# winner, and only then move on. Two phases:
#
#   1. A one-dimensional sweep per knob, always measured against the baseline,
#      which reveals each knob's individual effect and its best value.
#   2. Coordinate descent from that best-known point: re-sweep each knob while
#      the others are pinned at their current best, accepting only improvements.
#      This catches interactions a 1-D sweep cannot see (e.g. a large
#      max_batch_size only pays off once the queue delay is long enough).
#
# The proposer is deliberately stateless about results: campaign.py feeds each
# proposal, measures it, and reports the verdict back. That keeps the search
# logic testable without a live server.


def _sweep_variants() -> list:
    """Every single-knob variant, in a stable order."""
    variants = []
    for size in MAX_BATCH_SIZES:
        variants.append(Variant(max_batch_size=size))
    for name in PREFERRED_BATCH_SIZE_SETS:
        variants.append(Variant(preferred=name))
    for delay in QUEUE_DELAY_MICROSECONDS:
        variants.append(Variant(queue_delay_us=delay))
    for count in INSTANCE_COUNTS:
        variants.append(Variant(instance_count=count))
    return variants


def initial_queue(baseline: dict) -> list:
    """The ordered queue of variants a campaign should try, baseline first."""
    queue = [Variant()]  # the untouched baseline is experiment 01
    queue.extend(_sweep_variants())
    return queue


def neighbour_variants(current: Variant, tried: set) -> list:
    """Variants that change exactly one knob away from ``current``.

    ``tried`` holds Variant objects already measured (a Variant is frozen, so it
    hashes); anything already evaluated is skipped so descent terminates.
    """
    candidates = []
    for size in MAX_BATCH_SIZES:
        candidates.append(
            Variant(max_batch_size=size, preferred=current.preferred,
                    queue_delay_us=current.queue_delay_us, instance_count=current.instance_count)
        )
    for name in PREFERRED_BATCH_SIZE_SETS:
        candidates.append(
            Variant(max_batch_size=current.max_batch_size, preferred=name,
                    queue_delay_us=current.queue_delay_us, instance_count=current.instance_count)
        )
    for delay in QUEUE_DELAY_MICROSECONDS:
        candidates.append(
            Variant(max_batch_size=current.max_batch_size, preferred=current.preferred,
                    queue_delay_us=delay, instance_count=current.instance_count)
        )
    for count in INSTANCE_COUNTS:
        candidates.append(
            Variant(max_batch_size=current.max_batch_size, preferred=current.preferred,
                    queue_delay_us=current.queue_delay_us, instance_count=count)
        )
    return [c for c in candidates if c != current and c not in tried]


def next_phase(current: Variant) -> Variant:
    """The starting point for coordinate descent: the best single-knob point.

    campaign.py passes the best variant seen during the 1-D sweep; descent then
    explores combinations from there.
    """
    return current
