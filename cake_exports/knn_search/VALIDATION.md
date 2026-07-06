## Pre-publication GPU validation: PASS

- Hardware: `NVIDIA GB200` (`sm_100a`)
- Shapes: correctness `21/21`, CUPTI benchmark `11/11`; full generated suite `180` tests
- Validation shards: `1`; host wall time: correctness `155.07s`, benchmark `25.33s`
- `compute_speedup_vs_baseline` vs `flashlib.flash_knn`: min `1.2951x`, geomean `1.6593x`, max `1.8800x` (required minimum `1.0000x`)
- Measured: `2026-07-06T04:17:45+00:00`
- Full summary: [`VALIDATION.json`](VALIDATION.json); per-shape results: [`BENCHMARK_RESULTS.json`](BENCHMARK_RESULTS.json)
