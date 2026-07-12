## Pre-publication GPU validation: PASS — declared 11-shape performance floor

- Hardware: `NVIDIA GB200` (`sm_100a`)
- Shapes: correctness `198/198`, CUPTI benchmark `198/198`; full generated suite `206` tests
- Validation shards: `4`; host wall time: correctness `7.19s`, benchmark `11.92s`
- Full all-shape `compute_speedup_vs_baseline` diagnostic vs `flashlib.flash_knn`: min `1.4922x`, geomean `3.1141x`, median `3.0323x`, p90 `5.4709x`, max `8.2820x`; `0/198` shapes are below the nominal `1.0000x` threshold (diagnostic, not hidden; see `VALIDATION.json`).
- Publication performance floor: `11` explicitly named shapes; min `2.0085x`, geomean `2.9128x`, median `2.7616x`, p90 `4.1407x`, max `4.5096x` (required minimum `1.0000x`).
- Publication floor labels: `rag_q128_m131072_d128_k10`, `rag_q4096_m20000_d128_k10`, `rag_lowq_q8_m131072_d128_k10`, `rag_lowq_q16_m131072_d128_k10`, `rag_lowq_q32_m131072_d128_k10`, `rag_lowq_q64_m131072_d128_k10`, `ksweep_q4096_m20000_d128_k1`, `ksweep_q4096_m20000_d128_k5`, `ksweep_q4096_m20000_d128_k8`, `round54_q4096_m16384_d128_k8`, `round54_q4096_m32768_d128_k8`.
- Candidate lifecycle latency diagnostics: init-once median `559.2856 ms`; first-signature compute median/p90 `5.9713/7.9509 ms`; hot compute median/p90 `0.1284/0.5251 ms`

#### Hot steady-state synchronized E2E speedup

| Validated shape scope | Min | Geomean | Median | P90 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| All 198 benchmarked shapes (diagnostic scope) | 1.4922x | 3.1141x | 3.0323x | 5.4709x | 8.2820x |

#### Modeled after-init amortized synchronized E2E speedup

| Public calls N | Min | Geomean | Median | P90 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0747x | 6.4900x | 7.9356x | 12.4745x | 98.6406x |
| 10 | 0.5734x | 6.1971x | 6.9176x | 11.0294x | 85.2400x |
| 100 | 1.4063x | 4.4384x | 4.4430x | 7.4146x | 38.0379x |
| 1000 | 1.4832x | 3.3508x | 3.3140x | 5.7286x | 12.0086x |

#### Modeled including-init amortized synchronized E2E speedup

| Public calls N | Min | Geomean | Median | P90 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0006x | 0.0802x | 0.0941x | 0.2243x | 2.0554x |
| 10 | 0.0059x | 0.0991x | 0.1006x | 0.2386x | 2.0674x |
| 100 | 0.0575x | 0.2055x | 0.1741x | 0.3782x | 2.1709x |
| 1000 | 0.4645x | 0.8477x | 0.8440x | 1.2807x | 2.5708x |

- All three tables report synchronized host E2E speedups as `baseline/candidate`. `Hot steady-state` measures a repeated public call at each lane's declared hot cache state; its per-shape values supply the official metric used by the separate publication-floor section.
- `After-init amortized(N) = (first_compute + (N-1) * hot_median) / N`; it excludes init.
- `Including-init amortized(N) = (init + first_compute + (N-1) * hot_median) / N`; it includes init. Each latency formula is evaluated separately for baseline and candidate, then reported as `baseline/candidate`. Both amortized scenarios are composed from measured components, not a directly timed N-call loop. A lane without explicit init uses `I=0`.
- Init scope: `once_per_validation_shard_process_device_operator`; composition: `runtime_init_only`; baseline has explicit init: `no`.
- Cache policy: `synchronize_and_clear_after_each_completed_shape`; resident multi-shape cache benchmarked: `no`; cold order: `deterministic_balanced_per_publication_contract_portfolio`; init order: `candidate_only_baseline_has_no_explicit_init`
- Lifecycle timing convention: all three lifecycle tables are synchronized host E2E. Init/first-call brackets are CUPTI timestamp host diagnostics; separately, the hot GPU-span diagnostic remains strict correlated CUPTI activity timing.

- Measured: `2026-07-12T01:45:09+00:00`
- Full summary: [`VALIDATION.json`](VALIDATION.json); per-shape results: [`BENCHMARK_RESULTS.json`](BENCHMARK_RESULTS.json)
