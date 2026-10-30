# Progress log

Append-only. New entries go at the **bottom**; nothing above this line is ever
rewritten. Add an entry whenever a campaign finishes, a measurement settles an
open question, a bug changes how results should be read, or a Q&A establishes a
fact worth keeping. Prose over bullet-spam — the point is that a future reader
gets *why*, not just *what*.

---

## 2026-10-29 — Campaign #1, and the numbers are real (no caching)

### Campaign #1 result

25 experiments, 40.4 minutes, stopped by plateau ("6 consecutive experiments
failed to beat 2435.2 rps"). Baseline **650.75 rps** → adopted best **`ic4` at
2435.25 rps**, a **3.74x** speedup on the same GPU and the same model.

The highest number ever measured was `mbs64_ic4` at 2447.9 rps, which the ratchet
correctly rejected (+0.5%, under the 2% threshold). `analyze.py` reports adopted
and peak separately so this distinction survives — do not read the leaderboard
top row as "the winner".

### The finding that matters: batching barely helped, parallelism did everything

`instance_group.count` was the **only** knob that moved the needle:

| config | peak rps | gain over previous |
|---|---|---|
| `ic1` (baseline) | 650.8 | — |
| `ic2` | 1301.0 | +100% |
| `ic3` | 1881.1 | +45% |
| `ic4` | 2435.3 | +29% |

Everything else was flat or worse, all at `ic1`:

- `max_batch_size`: 445 (mbs1) … 638 (mbs64) — **every value below the 651 baseline**
- `max_queue_delay_microseconds`: 542 (qd0) … 686 (qd25000) — a ~6% band
- `preferred_batch_size`: 596–674, except `pbs:small` which **collapsed to 159**

So the README's "non-batched vs batched" framing needs revising. Going from
`mbs1` (445) to `mbs64` (638) is only 1.43x. Getting 3.74x took *more copies of
the model on the same GPU*, not bigger batches. The campaign plateaued on
`instance_count` rather than saturating it — it ran out of values at 4 while
still gaining 29% per doubling, which is why `INSTANCE_COUNTS` now includes 6 and
8.

Why batching underperforms is visible in the per-execution GPU time (exp-19,
`ic4`, from `stats.deltas.json`):

| connections | avg batch | ms / execution | ms / request |
|---|---|---|---|
| 1 | 1.00 | 0.84 | 0.84 |
| 6 | 2.00 | 1.41 | 0.70 |
| 21 | 4.73 | 3.53 | 0.75 |
| 32 | 7.20 | 5.53 | 0.77 |

A batch of 7.2 costs **6.6x** a batch of 1 — nearly linear. Dynamic batching is
buying ~11% compute efficiency. At 5 tokens this model is launch-overhead-bound,
not FLOP-bound, so there is almost no arithmetic to amortise. That is a property
of the toy model, not of Triton, and it is the strongest argument for testing a
real-sized embedding model.

### "Is the compute actually happening, or is it cached?"

Reasonable suspicion, because the harness sends a **byte-identical payload on
every request** — precisely the pattern a response cache exploits. Verified
**off**, four independent ways:

1. Triton's response cache needs **two** switches, and the docs state *"By
   default, no model uses response caching even if the response cache is enabled
   globally."* The server's args contain no `--cache-config` (and no deprecated
   `--response-cache-byte-size`).
2. The live `GET /v2/models/miniLM-onnx/config` contains no `response_cache`
   block, nor does `config.baseline.json`.
3. `GET /v2/models/miniLM-onnx/stats` reports `cache_hit.count: 0`.
4. **The decisive one, from our own data.** The stats schema is a built-in lie
   detector: `success.count` *includes* cache hits, `compute_infer.count`
   *excludes* them. At 32 connections in exp-19 both read **21795** — zero
   divergence, so zero hits, across every stage of all 25 experiments. And
   `execution_count: 3026` means the model was genuinely invoked 3026 times to
   produce those 21795 embeddings.

The GPU reading corroborates it. At 32 connections `compute_infer` accumulated
16.72 s over an 8 s stage on 4 instances = **52.3% duty cycle**, which matches
the "up to 50% GPU usage" seen in nvtop. A cache hit costs no GPU time at all;
2435 rps served from cache would read as ~0%.

### Where the 4 pegged cores come from (this is the real ceiling)

Measured with a controlled A/B at 32 connections, 25 s stages, sampling
per-core idle on the server (`top -bn2 -d1`, userspace column of a 16-core box):

| config | peak rps | server userspace | cores busy | ms CPU / request |
|---|---|---|---|---|
| `ic1` | 674 | 7.1% | ~1.14 | 1.69 |
| `ic4` | 2429 | 28.6% | ~4.58 | 1.89 |

Server CPU cost is **~1.8 ms per request and constant across configs** — linear
in *requests*, not in *executions*. Batching reduces GPU executions 7x but does
nothing for this, because every request still gets its own response containing
1920 FP32 values rendered as JSON **text**. So ~2435 rps x 1.89 ms ≈ **4.6
cores**: that is exactly the 4 cores pegging at 100%.

Two caveats on the measurement. `ps -o pcpu` is a *lifetime* average and showed
tritonserver at ~92% (≈1 core) — misleading for an intermittently-loaded
process; only the per-core sample is valid. And the client is a separate ~0.92
core (single-threaded asyncio parsing ~4.7M floats/second), which is not the
bottleneck yet but caps how much higher these numbers can honestly go.

**Consequence:** the next ceiling is JSON serialisation on the server, not the
GPU and not the network. Binary protocol is the fix, and it is the top item in
the README's TODO list.

### Network is not the bottleneck (settled)

~29 KB per request (293 B request + ~27 KB JSON response + framing). 12 MB/s ≈
410 rps averaged over a whole ramp including the idle gaps between experiments —
consistent with the ladder, not a ceiling. That is ~10% of gigabit. The clincher:
bytes per request are constant across all configs while throughput varied 445 →
2435 rps, so a network limit would have flattened every result to the same value.
If the 12 MB/s figure ever matters, check `ethtool <iface> | grep Speed` —
12 MB/s is suspiciously close to a 100 Mbps link's practical ceiling, which is
what a NIC that negotiated down would show.

### Bug found live, invisible to the mock

Every `max_batch_size` experiment initially failed with
`HTTP 500 failed to load 'miniLM-onnx', failed to poll from model repository`.
Triton rejects any config where a `preferred_batch_size` entry exceeds
`max_batch_size`, and the baseline preference list runs to 128 — so lowering
`max_batch_size` alone produces an **unloadable** config. The load API reports
every config rejection with that one generic string; the real reason appears only
in the container log (`docker logs <ctr> | grep "Poll failed"`).

Fixed in `experiment_config.py`: `normalize_config()` truncates the preference
list to legal values (omitting the field when nothing survives — an empty list is
also invalid), and `validate_config()` rejects illegal candidates locally, before
the server is touched. The mock never caught this because it never enforced the
invariant.

### Also learned

- `pbs:small` at 159 rps is a genuine collapse, not noise: preferred sizes
  `[2,4,8,16]` under a `max_batch_size` of 128 means the batcher refuses to
  launch anything until it can hit a preferred size, so batches stall at low
  concurrency. Worth remembering as the failure mode of an over-restrictive
  preference list.
- `qd0us` caps throughput at whatever a single launch can carry (2029 rps at
  batch 2.0 for `qd0us_ic4`) — with zero queue delay there is nothing to batch.
- The ratchet has no noise floor behind it yet. `pbs:mid` was accepted at +3.6%
  and `qd25000us` rejected at +0.98%, and we cannot yet say how much of either
  was run-to-run variance. Measuring that spread is the cheapest way to make
  every future verdict more trustworthy.

---

## 2026-10-30 — Per-stage telemetry, and it caught a 100 Mbps link immediately

### What was added

`system_metrics.py` samples **CPU per core**, **network utilisation against link
speed**, and **server VRAM** on both machines during every ramp stage, and
`ramp-up.py` prints a summary line under each stage and writes `system.json`
plus 12 new CSV columns. The server is sampled by piping one self-contained
Python program to `ssh 7700x python3 -`, which loops remotely and returns raw
jiffies and byte counters as JSON — one ssh handshake per stage rather than per
sample. Nothing in it can fail a run: every path degrades to
`{"available": false, "error": ...}`.

The point of the whole thing is attribution. A plateau at 620 rps and a plateau
at 620 rps because the NIC is full are the same number, and only the second one
means the config search is over.

### Headline: the client link is negotiating at 100 Mbps, not gigabit

The first instrumented run flagged it on the peak stage:

```
client: enx00e04c6803f3 at 98.1% of a 100 Mbps link
```

Throughput is now **flat at ~620 rps** across 8, 16, and 17 connections (618.2 /
622.1 / 621.8) with the wire pinned at 12.9 MB/s. The interface is a Realtek
**RTL8153 Gigabit** Ethernet Adapter (`lsusb`: `0bda:8153`, driver `r8152`)
reporting `speed=100`, `duplex=full`, `carrier=1`. A gigabit-capable part
negotiated down to fast ethernet.

A raw transfer over ssh confirms the pipe rather than the model: 200 MiB of
`/dev/zero` server-to-client took 18.07 s = **11.1 MB/s (88 Mbps)**, and the NIC
counter moved 209 MiB for 200 MiB sent, so the counters are honest.

**This overturns the answer I gave earlier in this project that the network was
not the bottleneck.** It was true when it was said; it is not true now.

### The link was gigabit during campaign #1 — verified false that it always was

Campaign #1's 2435.25 rps at ~19 kB of wire bytes per response (see the gzip
finding below) needs **374 Mbps**, which 100 Mbps cannot carry. So the
negotiation changed between 2026-10-29 and 2026-10-30 — cable, switch port, or
the adapter re-negotiating after a replug. Nothing in the repo changed.

The practical consequence: **campaign #2 must not be trusted until the link is
restored.** Every `ic6`/`ic8`/`ic10`/`ic12` result will be network-capped at
~620 rps and will look like a model-config plateau. Check the cable (Cat5e or
better, and all four pairs — 100BASE-TX works with only two), try a different
switch port, or force it with `ethtool -s enx00e04c6803f3 speed 1000 duplex full
autoneg on`.

### Why exactly 620 rps: gzip is on, and it is worth 2x

Responses are 38714 B of JSON at the application layer, but aiohttp sends
`Accept-Encoding: gzip` by default and Triton honours it:

```
Accept-Encoding='gzip'  ->  wire=18705  decoded=38714  ratio=0.483
```

Float-heavy JSON compresses roughly 2x. That resolved an anomaly that looked
like a broken counter for most of this session — app bytes (23.9 MB/s) were
almost exactly 2x NIC bytes (12.2 MB/s) at high rate while agreeing at low rate.
Both were correct; one is post-decompression, the other is the wire.

With 18705 B response + 293 B request + ~200 B headers = ~19.2 kB per request:

| link | ceiling | measured |
|---|---|---|
| 100 Mbps | 651 rps | **622 rps = 95.5% of it** |
| 1000 Mbps | 6511 rps | — |

The plateau is the wire, to within 5%. Uncompressed the same link would cap at
319 rps, so gzip is currently worth about 2x and is *not* a cost worth
removing. Note this also means the binary-protocol item in the README gets
*less* urgent than it looked, not more: binary output would remove the JSON
text before compression, but gzip already recovers most of that on the wire.
Binary's remaining win is server CPU, not bytes.

### Client is not the ceiling (yet), and a classifier bug fixed on the way

At the plateau the client's busiest core averages **76%**, aggregate 24%; the
server is at 8% aggregate with one core at 64%. Neither is pinned, so the
620 rps is not the harness. The earlier `cores_at_95pct` used per-core *max*,
which fired "1 core(s) at >=95%" on a transient spike under 8% aggregate load —
a false bottleneck claim. `classify()` now uses a new
`cores_sustained_95pct` (per-core *mean* >= 95, i.e. pinned for the whole
stage); the max-based count is still recorded for the CSV. Watch the client's
busiest core on the next gigabit run — at 2400 rps it may well pin, which is the
open README question.

### Triton thread defaults, from the core-count question

No flag sets a core count. From `src/command_line_parser.h`:
`http_thread_cnt_{8}`, `buffer_manager_thread_count 0`,
`model_load_thread_count 4`, `grpc-infer-thread-count 2`,
`sagemaker-thread-count 8`. `--host-policy=<n>,cpu-cores=` only applies to
instances that declare that host policy, so it would break the load-API configs.
To actually cap Triton, use docker `--cpuset-cpus=0-7`. Cross-validated the
sampler against `top` on the server: 0.4% vs 0.5% idle. Worth remembering that
`--http-thread-count` defaults to **8**, which may itself cap an `ic8` sweep —
raise it to 16 before blaming the model for an `ic8` plateau.

---

## 2026-10-30 — `--http-thread-count=16` added to `run_triton.sh`

Follow-up to the thread-default finding above. `triton-models/run_triton.sh` now
passes `--http-thread-count=16` (Triton's default is 8). Each in-flight HTTP
request occupies one thread from that pool, so at the default the server cannot
have more than 8 requests being read, parsed, and written at once regardless of
how many model instances are loaded — which would have capped the
`instance_group.count` sweep at exactly the point it gets interesting (ic8+) and
looked like a model plateau. 16 matches the server's core count.

Reminder recorded in the script's header comment: this raises a **thread** count,
not a core count. Triton has no core-count flag; the real cap is docker
`--cpuset-cpus=0-7`.

Takes effect on the next `run_triton.sh` restart, which means a server restart —
so it must be done before campaign #2, not during it. Still gated on restoring
the gigabit link.
