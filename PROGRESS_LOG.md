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

---

## 2026-11-01 — BERT config fixed, and embeddings without a Python wrapper

Two things landed here that are easy to lose: why the original config was
broken, and why the embeddings path ended up being graph surgery rather than a
Python wrapper.

### The config bug, and the trick that found it

`bert-base-uncased` failed to load with `tensor 'logits': the model expects 3
dimensions (shape [-1,-1,30522]) but the model configuration specifies 2
dimensions`. The cause is a rule that is easy to forget: when
`max_batch_size > 0`, the config's `dims` **excludes** the batch dimension. So
`dims: [ -1 ]` was being read as complete shape `[-1,-1]`, not `[-1,-1,-1]`.
Fix was `output.dims: [ -1 ]` → `[ -1, 30522 ]`.

Worth keeping is *how* the real signature was obtained. `onnx` and
`onnxruntime` are both unimportable in the container and in `/AI/triton/my_env`,
so reading the graph by hand was not available. Instead: bind-mount **only** the
`1/` version directory into a scratch model repo with **no** `config.pbtxt`, let
Triton auto-complete the config from the model itself, then
`GET /v2/models/<name>/config`. Triton reports the authoritative answer. Far
more reliable than guessing dims from docs.

Two environment traps found on the way: symlinks inside a bind-mounted model
repo break with a misleading `failed to stat file /opt/ml/model/...` because the
link target does not exist inside the container — bind-mount the real directory.
And a `vim config.pbtxt` left open on the host holds the pre-fix buffer, so a
later `:w` silently reverts the fix.

### Embeddings: trimmed the ONNX graph instead of wrapping it

The question was whether to switch to safetensors plus a Python wrapper to get
embeddings out. Neither, and the reasons were measured rather than assumed:
there are **no** bert-base-uncased safetensors anywhere on the host, and the
Python wrapper path costs a **3.0 GB** `minilm_env.tar.gz` execution environment
per model.

Graph inspection showed the model is `BertForMaskedLM` (1266 nodes, opset 11,
204 initializers, no pooler node), ending at
`/cls/predictions/decoder/MatMul`. So the waste is the 30522-way vocab head, not
where pooling happens. `trim_to_embeddings.py` re-anchors the graph output to
`last_hidden_state` (`/bert/encoder/layer.11/output/LayerNorm/Add_1_output_0`)
and appends attention-masked mean pooling in-graph.

Deployed as a **separate** model (`bert-base-uncased-embed`) so the working MLM
config stayed untouched, then hot-loaded through the explicit-mode load API with
no server restart. Validated against a PyTorch reference:

| check | result |
|---|---|
| max abs error vs PyTorch | 1.669e-06 |
| padding invariance | 0.000e+00 |
| MLM nodes remaining | 0 |
| response size at seq 128 | ~350x smaller |

The padding-invariance check is the one that matters. A mean-pooling node that
ignores `attention_mask` produces plausible embeddings for a single unpadded
request and quietly wrong ones for any padded batch — which is exactly the case a
throughput harness creates. That same class of bug reappears below, so it is
worth treating "verify padding invariance" as a standing gate for this project.

---

## 2026-11-01 — intent-replica: the corpus lives server-side, and two silent bugs

The harness up to this point benchmarked **only the encoder**. The workload being
reproduced at work is BERT encode + cosine similarity against a corpus + return
the matched corpus string, so `triton-models/intent-replica/` now reproduces that
shape. Design decisions that came from the user and should not be relitigated:
the corpus is **server-side model state and never appears in a request**; a
request is one user utterance and the response is the matched line; cosine is a
**numpy dot product on CPU** because that is what production does; the encoder is
**HF torch in the python backend** for the same reason.

Corpus built from banking77 (train+test) and CLINC150 (`imbalanced`
train/validation/test + `plus` train):

| property | value |
|---|---|
| unique utterances | 36,912 |
| distinct intents | 227 |
| `corpus.npy` | 36,912 x 768 float32, 113.4 MB |
| vector norms | min 1.000000, max 1.000000 |
| exact self-match | 99.98% |
| median runner-up similarity | 0.9162 |
| encode rate (3080) | ~2,500 utt/s |

Because vectors are stored unit-length, serving cosine is a bare `corpus @ q`
with no normalisation on the hot path.

### Bug 1: an embedded newline silently misaligned the corpus

`corpus.txt` had **36,914 lines for 36,913 vectors**. One banking77 utterance
contains a literal newline, so a line-oriented index file gained a row and every
subsequent row pointed at the wrong vector and label. Nothing raised. The
argmax would still have returned a confident, plausible, wrong string.

Fixed at the root by normalising whitespace (`" ".join(text.split())`) during
dedupe, which also merged a whitespace-only duplicate — hence 36,913 → 36,912.
Backstopped twice: the builder re-reads its own output and asserts one line per
vector, and `initialize()` refuses to load unless
`len(vectors) == len(texts) == len(labels)`.

### Bug 2: mean pooling let padding leak in

Self-match scored **0.926**, not 1.0. The pooling was
`hidden.sum(1) / mask.sum(1)`, but BERT's hidden states at pad positions are
**non-zero**, so unmasked `sum` adds garbage and only the *count* is masked. The
error scales with how much padding the batch contains, so build-time (256-row
batches) and serving-time (1-row requests) drifted into **different vector
spaces**. Correct form is `(hidden * mask).sum(1) / mask.sum(1)`.

The lesson is in how it was caught. The existing self-check compared batched
vectors against each other, so it could never see this — both sides had the same
padding error. The check that catches it re-encodes a sample **one row at a
time**, which is what a single-row request does:

```
single-row vs batched agreement min=1.000000 median=1.000000
```

After the fix, `./test-intent.sh --check` returns the input line at score
exactly 1.0. Standing rule for this project: **a retrieval model that returns
plausible strings and plausible latency is not verified.** Assert a known-input
self-match at 1.0 before trusting any throughput number built on it, because a
silent argmax or alignment bug is invisible in latency data.

### Cost of the server-side corpus, stated up front

Each Triton python-backend instance is its own process, so the 113 MB corpus is
multiplied by `instance_group.count` and cold start grows with it. Do not read a
slow `POST /v2/models/intent-replica/load` as a hang. This is the same scaling
production pays, so it belongs in the report rather than being optimised away.

`minilm_env.tar.gz` is reused via `EXECUTION_ENV_PATH` pointing at the **other**
model's copy. Legal — it is just a file path — and saves a second 3 GB
environment.

Container paths trip this up more than once. The whole model repository is
bind-mounted, so host `/AI/models/triton` maps to container `/opt/ml/model` and
a model's files land at `/opt/ml/model/<model>/<version>/`. The corpus is
therefore `/opt/ml/model/intent-replica/1/corpus.npy`, **not**
`/opt/ml/model/corpus.npy`. `docker exec triton find / -name corpus.npy` settles
a path question in a few seconds and beats guessing — worth reaching for before
writing a script that loads a file by hand.

---

## 2026-11-01 — GPU utilisation measured, and the batching I had reported was fake

Asked two questions about the numbers above: what was the GPU utilisation, and
did the sweep include batching alongside instance counts. The honest answers were
"not measured" and "no", so both were measured properly. Doing it properly
overturned a conclusion I had already reported.

### Reading batch size from Triton, not from config

`config.pbtxt` says `dynamic_batching` is on. That is not evidence batching
happens, and not evidence it *helps*. The real number is
`nv_inference_request_success / nv_inference_exec_count` from `:8002/metrics` —
how many requests the scheduler actually grouped per `execute()` call. Sampled
with `nvidia-smi dmon -s um` during load by `tools/measure.sh`, which reports
throughput, GPU utilisation and batch size together so a future run cannot
repeat this mistake.

At conc=16 the ratio was **about six** (5.6–6.0 across repeat runs): Triton was
happily grouping six requests per call. But `execute()` was written
`for request in requests:` — the faithful production shape — so it looped over
the group and did the encoder call and the full corpus read **once per request**.
The scheduler's batching was real and completely wasted. Every throughput number
reported before this point was measuring the unbatched path regardless of what
the config asked for.

### Batching is the dominant lever, not the cosine device

`BATCH_ACROSS_REQUESTS` (default `false`, preserving the faithful production
comparison) encodes and retrieves the whole group at once: N corpus reads and N
encoder launches become one of each. Verified to return **byte-identical** scores
to the per-request path before it was measured.

RTX 3080, conc=16, 20s windows, identical payloads. Every row below was
re-measured from a cold reload with the loaded config confirmed via
`GET /v2/models/intent-replica/config`:

| cosine | batch | instances | rps | p50 | GPU sm avg (max) | GPU mem avg |
|---|---|---|---|---|---|---|
| cpu | off | 1 | 157 | 99.7 ms | **21%** (28%) | 8% |
| cpu | off | 2 | 125 | 126.7 ms | 21% (33%) | 7% |
| cpu | on | 1 | 373 | 41.4 ms | 11% (18%) | 4% |
| cuda | off | 1 | 422 | 37.7 ms | **77%** (97%) | 29% |
| cuda | off | 2 | 426 | 37.5 ms | **77%** (97%) | 29% |
| cuda | on | 1 | **1677** | **9.5 ms** | 41% (52%) | 18% |
| cuda | on | 2 | 1392 | 12.0 ms | 70% (87%) | 23% |

Pushing concurrency on the best config: conc=32 gives 2766 rps at 14.8 rows/exec
and 47% sm; conc=64 gives **3754 rps** at 28.2 rows/exec with GPU sm still only
**52%**.

Two rows in the first version of this table were wrong and are corrected above.
`cuda / off / 1` was written as 314 rps at 44% sm; it is actually **422 rps at
77% sm**. `cuda / off / 2` was written as 419 rps; it is 426. I cannot now
reconstruct exactly which measurement produced the old pair — the sweep that
produced them predated `tools/measure.sh`, and the `sed`-against-a-container-path
trap below was active during part of it, which is enough to make any row from
that run untrustworthy. The corrected numbers change two of the conclusions,
which is why they are recorded rather than quietly fixed: **if a sweep was run
before the harness reported the loaded config alongside the result, re-run it
before citing it.**

`rows/exec` is the noisiest column in this data (5.63 vs 5.98 across identical
runs). Treat it as "roughly six" rather than a precise quantity: it is a
diagnostic for *whether* the scheduler is grouping requests, not a metric to
optimise. Throughput and utilisation were stable to within a percent or two.

Four conclusions, two of which correct statements I made earlier:

1. **The production shape reproduces the low utilisation under investigation:
   21% SM, 8% GPU memory, 157 rps.** That is the number the GPU-spend question
   turns on, and it is reproduced with the encoder untouched.
2. **Batching is the dominant lever; GPU cosine is the second one.** Best config
   versus the production shape is 157 → 1677 rps, a **10.7x** combined gain, but
   that conflates two changes. Isolating each at fixed conditions: batching alone
   is 2.4x on CPU cosine (157 → 373) and **4.0x** on GPU cosine (422 → 1677),
   while moving cosine to the GPU alone is 2.7x (157 → 422). I had presented the
   CPU-vs-GPU cosine A/B as the headline; it is the smaller lever. Note also that
   batching helps *more* once cosine is on the GPU — with cosine still on CPU the
   batched corpus read is the limit, since 373 rps x 113 MB is about 42 GB/s of
   host memory bandwidth.
3. **A second instance is worth nothing here, and costs memory.** With GPU cosine
   unbatched, 1 → 2 instances is a tie (422 → 426 rps) while framebuffer use
   climbs. Once batching is on it is actively worse (1677 → 1392 rps), because
   one batched instance already keeps the GPU fed and the second process
   duplicates the 113 MB corpus. My earlier "2 instances is the knee" was an
   artefact of measuring the unbatched path.
4. **High GPU utilisation is not the goal, and chasing it is a trap.** The two
   `cuda / off` rows saturate the GPU at **77% avg, 97% max** and deliver ~424
   rps. The batched single instance runs at **41% sm** and delivers **1677 rps** —
   four times the throughput on half the utilisation. Utilisation measures how
   busy the SMs are, not how much useful work completes per request. Since the
   question behind this project is whether the employer is over-spending on GPUs,
   say it plainly: the production shape sits at 21% sm because each request does
   its own corpus read and encoder launch, and the fix (batching) *lowers*
   reported utilisation while multiplying throughput. Anyone tuning toward a
   greener utilisation graph will pick the 4x-slower config.

### Reconciling with campaign #1, which said batching barely helped

Campaign #1 found `max_batch_size` flat-to-worse and concluded batching bought
~11% compute efficiency, with `instance_group.count` doing all the work. That
still stands for the model it measured, and it is not in conflict with the 4.0x
above — the two runs are measuring different bottlenecks.

Campaign #1 drove `miniLM-onnx` at `TOKEN_COUNT = 5`, and its own per-execution
table shows why: a batch of 7.2 cost 6.6x a batch of 1, i.e. nearly linear,
because at 5 tokens that model is **launch-overhead-bound with almost no
arithmetic to amortise**. There is nothing for a batch to share.

`intent-replica` has a large fixed per-request cost that a batch *can* share: the
113 MB corpus read and a fresh encoder launch per request. Batching amortises
exactly that. So the project rule is: **batching helps in proportion to the
per-request fixed cost, and does nothing for a small launch-bound model.** A
sweep result about `max_batch_size` is a statement about the model's cost
structure, not a general property of Triton — do not carry a "batching doesn't
help" conclusion from one model to the next.

The same caution runs the other way. Campaign #1 concluded
`instance_group.count` was the only knob that mattered, and on `intent-replica`
a second instance is a tie unbatched and a 17% loss batched. Instance count
helps when one instance cannot keep the device busy; it stops helping, and
starts costing duplicated model state, as soon as it can. Both campaigns are
correct about their own model and neither result transfers.

### What would mislead a reader of these numbers

- Payloads rotate through 5 short utterances, so the encoder sees unrealistically
  uniform input. Query length 10-25 tokens is the highest-value remaining sweep
  precisely because the encoder, not retrieval, is now the ceiling.
- The earlier hypothesis that CPU cosine is memory-bandwidth-bound (113 MB/request
  ≈ 17 GB/s at 157 rps) does explain the production shape's 21% utilisation, but
  it is **not the binding constraint once batching is on**. At 3754 rps with GPU at
  52%, the encoder is the ceiling.
- Config on the host is left at the faithful production default: `cpu`, batch off,
  1 instance.

### Measurement traps, three of which cost real time

These are worth keeping together because each one produces *confident, wrong*
output rather than an error.

- **`nvidia-smi dmon -s um` column order.** It is
  `gpu sm mem enc dec jpg ofa fb bar1 ccpm`, so sm is `$2`, mem is `$3`, fb is
  `$8`, and the header occupies two lines. An awk filter keyed on the wrong column
  silently samples idle rows.
- **awk compares strings unless you force numbers.** `if (sm > max)` where both
  look numeric does a *string* comparison, so `"9" > "22"` and every `max` in the
  report is wrong — it can come out too high or too low depending on the digits
  seen, which is what makes it so hard to notice. Write `sm=$2+0` first.
- **Never `sed` a container path from the host.** `sed -i /opt/ml/model/...`
  operates on a host path that does not exist, fails, and leaves the config
  unchanged — so a whole sweep comes back suspiciously uniform. Either `docker
  exec sed` or edit the host path under `/AI/models/triton/`. It recurred while
  writing this entry: a `sed` pattern written against `KEY { string_value: ... }`
  matched nothing because keys and values are on separate lines in
  `config.pbtxt`. The guard is to `grep` the file after editing, and better still
  to confirm the *loaded* config with
  `GET /v2/models/<name>/config`, which reports what the server is actually
  running rather than what the file says.

---

## 2026-11-01 — Beads sync is git-only; `bd dolt push` was never going to work

Removed `bd dolt push` from the session-close workflow in `AGENTS.md`. Worth
recording because the reason is not "the remote is misconfigured" — the command
is simply not part of how this repo syncs, and bd's own templates keep telling
agents to run it anyway.

`.beads/issues.jsonl` is tracked by git, and a `pre-commit` hook (installed by
bd, marked "managed by beads") exports the database into it and stages it on
every commit. So `git push` alone carries beads data. `bd dolt push` prints a
"supported remote URLs" usage error and exits 0, because the configured remote is
still the placeholder `git+ssh://git@github.com/org/repo.git`. Exit 0 is why this
went unnoticed for so long: the mandated session-close step appeared to pass.

Two things that would mislead a reader here:

- **The suggestion is structural, not a mistake to fix once.** `bd prime`,
  `bd onboard`, and `bd setup --print` all inject `bd dolt push` from stock
  templates. `AGENTS.md`'s `BEADS INTEGRATION` block is generated by four
  separate recipes (`codex`, `opencode`, `mux`, `factory` all target that file),
  so deleting the line inside the block does not survive a regeneration. The
  override therefore lives in a "Beads Sync Model" section *outside* the managed
  markers, and says plainly not to run it.
- **The JSONL is an export, not the working store.** `metadata.json` reports
  `backend: dolt, dolt_mode: embedded`, so `bd` reads and writes
  `.beads/embeddeddolt/` and regenerates the JSONL. Hand-editing the JSONL looks
  like the flat-file workflow but gets overwritten by the next `bd` write. Edit
  issues with `bd`; treat the JSONL as a build artifact that happens to be
  committed.

`CLAUDE.md` carries the same generated block and still contains the `bd dolt
push` step, so an agent reading that file instead is still told to run it.
