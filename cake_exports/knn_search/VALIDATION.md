## Pre-publication GPU validation: PASS

- Hardware: `NVIDIA B200` (`sm_100a`)
- Shapes: correctness `21/21`, CUPTI benchmark `11/11`; full generated suite `180` tests
- Validation shards: `1`; host wall time: correctness `157.40s`, benchmark `23.02s`
- `compute_speedup_vs_baseline` vs `flashlib.flash_knn`: min `1.2678x`, geomean `1.7826x`, max `2.3938x` (required minimum `1.0000x`)
- Runtime lifecycle: `runtime_init_only`, init once per validation shard, candidate init median `233.5249 ms`; first-signature compute median/p90 `47.5679/102.9273 ms`; hot public compute median/p90 `0.1700/0.2395 ms`
- Hot public E2E speedup: min `1.2678x`, geomean `1.7826x`, median `1.7978x`, p90 `2.2580x`, max `2.3938x`
- Modeled public-call geomean speedups (`baseline/candidate`): N=1: after-init 69.4441x/ including-init 4.8845x, N=10: after-init 58.2719x/ including-init 4.8681x, N=100: after-init 28.9727x/ including-init 4.7146x, N=1000: after-init 8.0427x/ including-init 3.7823x
- Cache policy: `synchronize_and_clear_after_each_completed_shape`; resident multi-shape cache benchmarked: `no`; cold order: `deterministic_balanced_per_publication_contract_portfolio`; init order: `candidate_only_baseline_has_no_explicit_init`
- Lifecycle convention: init/first-call brackets are CUPTI timestamp host diagnostics; steady GPU span remains strict correlated CUPTI activity timing.

- Measured: `2026-07-06T08:26:54+00:00`
- Full summary: [`VALIDATION.json`](VALIDATION.json); per-shape results: [`BENCHMARK_RESULTS.json`](BENCHMARK_RESULTS.json)
