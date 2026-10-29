# program.md — instructions for whoever (or whatever) runs the next experiment

This is the human-editable protocol, mirroring `program.md` in
`karpathy/autoresearch`. The driver in `campaign.py` implements exactly what is
written here. If you change a rule below, change the code to match — the
document is the contract.

## Hypothesis

The serving configs in use are badly under-tuned. On this RTX 3080, the same
models that currently need expensive GPUs should reach roughly **10x** the
throughput of a naive non-batched configuration purely by getting
`dynamic_batching` right. The campaign's job is to find that configuration and
prove the multiplier with numbers.

Baseline measured so far, both models loaded, current production config
(`max_batch_size 128`, `preferred_batch_size [2..128]`, queue delay 5000 µs):

| run | stages | concurrency | peak aggregate RPS | peak at |
|---|---|---|---|---|
| `ramp-results/` | 13 | 1..61 step 5 | 1152.2 | 51 |
| `results-32/` | 8 | 1..36 step 5 | 777.5 | 26 |

Both peaked and then **collapsed** (1152→826, 778→715) while p99 blew out. The
collapse point is itself a result worth charting, not just the peak.

## The one-metric rule

Every experiment is judged on **peak aggregate RPS** and nothing else. Keep a
config only if it beats the incumbent by more than **2%**. Everything else —
latency, batch size, failure rate — is *diagnostic*, recorded but never used to
justify keeping a slower config.

Secondary, for tie-breaking only: **RPS at 8 / 16 / 32 connections**. Peak at
20 connections beats peak at 100.

## Fixed vs variable

**Never varies within a campaign.** If you change one of these, you have started
a new campaign and every previous number is void:

- ramp schedule, stage duration, warmup
- request payload and endpoint
- the Triton server and its container image
- plateau and failure stop thresholds

**Varies, one knob at a time:** `max_batch_size`, `preferred_batch_size`,
`max_queue_delay_microseconds`, `instance_group.count`.

## Folder contract

`experiments/<campaign-ts>/exp-<NN>_<peakRps>rps/` must contain:

```
config.baseline.json  config.sent.json  config.applied.json  config.diff.json
ramp-results.csv      stats.deltas.json run.json             stdout.log
ramp-*.png
```

The campaign folder must contain `campaign.json`, `results.tsv`, `progress.png`.
Never delete a folder, including failures.

## Hard safety rules

1. **Always restore the baseline config** — at campaign end, on Ctrl-C, and on
   crash. The server must be left as it was found.
2. **Never hand-write a config.** `GET /config`, deep-copy, mutate one knob,
   `POST /load`. The load API does a *full replacement*, so a partial config
   silently deletes required fields.
3. **Verify every apply** with a follow-up `GET /config`. On mismatch, mark the
   experiment invalid and move on. Never measure an unverified config.
4. **Unload is asynchronous.** Poll `POST /v2/repository/index` until
   `state == "UNAVAILABLE"` **and** `reason == "unloaded"`. Do not use
   `/v2/models/{name}/ready` — it is also false during `UNLOADING`, and a reload
   started then races the old instance for GPU memory.
5. **Never unload both models at once.** Only the model under test is touched.
6. **Stats are cumulative since load and cannot be reset.** Always diff
   before/after. They are also written *after* the response returns, so the
   "after" snapshot must settle before it is read.
7. **`preferred_batch_size` entries must all be <= `max_batch_size`.** Triton
   refuses to load a config that breaks this, so lowering `max_batch_size` on
   its own produces an unloadable config — the preference list has to be
   truncated in the same mutation. `apply_variant` calls `normalize_config` to
   do this, and `validate_config` rejects anything still illegal *before* the
   server is touched. This matters more than it looks: the load API reports
   every config rejection as the same opaque
   `HTTP 500 failed to load 'X', failed to poll from model repository`, and the
   real reason only appears in the server log (`docker logs <ctr> | grep
   "Poll failed"`). Never conclude from that HTTP message that the server is
   broken — check the config first.

## Interpreting results

- **Realised avg batch size** (`inference_count / execution_count`) is the
  mechanism. A config that wins without moving batch size won by noise.
- `max_batch_size: 1` is the control: it must produce avg batch size exactly
  1.0. If it does not, the harness is broken, not the model.
- A large `max_batch_size` with a short queue delay batches poorly — the two
  knobs interact, which is why descent follows the sweep.
- `instance_group.count > 1` multiplies VRAM. Fine at 86 MB on a 10 GB card, but
  watch for it if a larger model is introduced.

## How to extend

**Add a search value:** edit the lists in `experiment_config.py`. Nothing else.

**Add a dimension:** implement it in `apply_variant`, `config_fingerprint`,
`config_diff`, and `neighbour_variants`. The sweep, descent, ratchet, folders,
and report all pick it up automatically.

**Add a campaign type** (e.g. the priority test): the README asks for holding
`miniLM` at 100 RPS while stepping the other model upward. That needs open-loop
load — a fixed offered rate — which the closed-loop ramp cannot express. Add
`--mode open-loop --target-rps` to `ramp-up.py` and reuse the same folder
contract and ratchet.

## Out of scope for this campaign

Swapping in a larger embedding model, and comparing against the SageMaker
`serve` wrapper. Both change the server or model set, which breaks the fixed
schedule that makes runs comparable. Run them as separate campaigns.
