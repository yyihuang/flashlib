## Pre-publication GPU validation: PASS

- Hardware: `NVIDIA B200` (`sm_100a`)
- Shapes: correctness `228/228`, CUPTI benchmark `228/228`
- Speedup vs `triton_h200_07cf`: min `0.1103x`, geomean `0.8642x`, max `12.7360x`
- Aggregate latency speedup: `1.0574x`; per-shape wins/losses: `129/99`
- Measured: `2026-07-04T07:43:00+00:00`
- Full summary: [`VALIDATION.json`](VALIDATION.json); per-shape results: [`BENCHMARK_RESULTS.json`](BENCHMARK_RESULTS.json)
