## Plan: Triton Ramp-Up Benchmark

Add a standalone [ramp-up.py](ramp-up.py) that reuses the request payloads in [benchy.py](benchy.py), maintains persistent connections to both Triton endpoints, and writes throughput and latency graphs as PNGs. No web app is needed.

**Steps**
1. Define adjustable constants for the SageMaker host, stage duration (10 seconds), starting pool size (1), increment (5), maximum (100), and early-stop thresholds.
2. Use `aiohttp` workers that continuously send requests through a shared keep-alive session. Ramp both endpoints in lockstep; shorten the final increment so the pool reaches exactly 100 after 96.
3. Record per-stage successful throughput, mean and p95 latency, and failure rate for each endpoint. Stop if a stage reaches a 10% failure rate or aggregate successful throughput improves by less than 2% for two consecutive stages.
4. Print a summary and generate Matplotlib PNG line graphs for throughput and latency. Matplotlib will be the only new dependency.
5. Verify the ramp boundaries, thresholds, metrics, and graph output, then run a short smoke test against Triton if available.

**Relevant files**
- [benchy.py](benchy.py) — endpoint and payload patterns to reuse.
- [test-onnx.sh](test-onnx.sh) — SageMaker host and ONNX target-model header convention.
- [test-sage.sh](test-sage.sh) — SageMaker-hosted Python-model request example.
- [ramp-up.py](ramp-up.py) — new standalone benchmark.

**Decisions**
- Default to `192.168.1.151:8000`; apply the target-model header to the ONNX endpoint as shown in `test-onnx.sh`. Keep routing settings editable.
- Interpret pool size as concurrent workers per endpoint, each issuing requests continuously over reusable connections.
- Save results locally as PNGs; no web UI or changes to the existing benchmark scripts.

This plan is ready for your review.