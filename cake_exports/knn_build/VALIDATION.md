# B200 Export Validation

- Contract rows: 112/112; unique shapes: 112
- Candidate correctness: 112/112
- Live `flashlib.flash_knn` correctness: 112/112
- Expected route parity: 112/112
- Baseline/public/prepared session provenance: 112/112
- Exact Cake source: `d524984607b363fb057a835c247e4f40f4de8e47` (!250)
- Prepared speedup vs live baseline: min 1.2299x, geomean 1.8788x, median 1.7191x, max 6.1468x
- Prepared rows below the required 1.2x floor: 0/112
- Public one-shot speedup vs live baseline: min 0.2261x, geomean 0.5293x, max 1.9120x
- Generated package tests: 124/124 passed
- Timing: CUPTI correlated GPU span with cold-L2 flushing on NVIDIA B200

The public and prepared host/enqueue, synchronized E2E, cold-first-call, GPU
span, kernel-sum, active-union, inter-kernel-gap, and distinct correlated
launch/kernel activity fields are recorded for every row in
[`BENCHMARK_RESULTS.json`](BENCHMARK_RESULTS.json).

The public one-shot path includes norm computation and launch/scratch
preparation on every call. The prepared path reuses a fully marshalled,
stream-bound sequence and is the steady-state performance entry point.
