
## How this was built
Built the plan with free copilot credit - luna 6 was chosen by 'auto' in plan mode, local.
Manually switched to Radiance qwen3.8 flash next to build the plan. 
Astonishingly fast. 1-7k prompt processing, 150-200 t/s Feels significantly faster than Luna.

## Purpose
I want a way to repeatably test the latency and throughput of a triton server with more than one model being served.  The goal is to iterate a bit on batch configurations with regard to size, wait duration, etc. and see what sort of throughput we can achieve on an RTX3080 which is a good stand in for Ampre GPU in EC2 level of performance.

I would like to get a non batched vs a batched configuration comparison in place to prove out the hypothesis thatr we are spending 10x what we need on GPU's at the moment.

## Running a benchmark

```bash
# one-shot ramp: 1 -> 100 connections, +5 per stage, 10s per stage
python ramp-up.py

# a shorter, single-model run written somewhere specific
python ramp-up.py --only onnx --max 60 --duration 8 --output-dir /tmp/quick
```

Every knob is a flag; `python ramp-up.py --help` lists them. Each run writes a
CSV, PNG graphs, `run.json` (provenance) and `stats.deltas.json` (Triton's own
server-side counters, including realised average batch size).

## Running a config-search campaign

`campaign.py` iterates model configurations autonomously — apply, verify,
benchmark, keep or discard — and stops on plateau, experiment cap, or time
budget. See `EXPERIMENT_CAMPAIGN_PLAN.md` for the design and `program.md` for
the protocol.

**One-time prerequisite:** restart Triton with
`triton-models/run_triton.sh`, which now passes `--model-control-mode=explicit`
so configs can be changed at runtime. Without it the campaign refuses to start.

```bash
python campaign.py --dry-run                        # show the search order only
python campaign.py --max-experiments 6 --budget 900 # a short real campaign
python analyze.py                                   # leaderboard + progress.png
```

Results land in `experiments/<campaign-timestamp>/exp-<NN>_<peakRps>rps/`.

## TODO things to test out

Covered by `campaign.py` (it sweeps these automatically):
- `max_batch_size` — swept over 1, 8, 16, 32, 64, 128. `1` is the non-batched
  control the hypothesis needs, and must report avg batch size 1.0.
- `preferred_batch_size` — swept over several subsets plus "auto" (field omitted).
- `max_queue_delay_microseconds` — swept over 0, 500, 2000, 5000, 10000, 25000.
- `instance_group.count` — swept over 1..4. Yes, each count is another full copy
  of the model in VRAM (86 MB here, fine on a 10 GB 3080). The campaign records
  realised batch size so you can see whether more copies beat a longer queue
  delay rather than guessing.

Still to do:
- Adjust the docker run command to use the SageMaker `serve` wrapper instead of
  Triton directly. This changes the server, so it has to be its own campaign.
- Serve a larger embedding model than miniLM-L6-v2. Also a separate campaign,
  for the same reason.
- Try a single model instead of two at a time. Use `campaign.py --model
  miniLM-onnx --ramp-endpoint onnx`, which already benchmarks one model in
  isolation; unloading the other is a small addition if the noise matters.
- Hold the priority-1 model at 100 requests/second while increasing the lower
  priority model, and check the priority-1 p95 stays flat. This needs open-loop
  load (a fixed offered rate), which the closed-loop ramp cannot express — see
  the "Add a campaign type" note in `program.md`.