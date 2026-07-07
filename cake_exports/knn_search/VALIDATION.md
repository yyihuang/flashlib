## Pre-publication GPU validation: PASS — declared 11-shape performance floor

- Hardware: `NVIDIA GB200` (`sm_100a`)
- Shapes: correctness `198/198`, CUPTI benchmark `198/198`; full generated suite `206` tests
- Validation shards: `4`; host wall time: correctness `235.31s`, benchmark `34.17s`
- Full all-shape `compute_speedup_vs_baseline` diagnostic vs `flashlib.flash_knn`: min `0.0105x`, geomean `1.0556x`, median `1.5706x`, p90 `2.0658x`, max `4.1821x`; `53/198` shapes are below the nominal `1.0000x` threshold (diagnostic, not hidden; see `VALIDATION.json`).
- Publication performance floor: `11` explicitly named shapes; min `1.4046x`, geomean `1.6893x`, median `1.6572x`, p90 `1.8872x`, max `1.9308x` (required minimum `1.0000x`).
- Publication floor labels: `rag_q128_m131072_d128_k10`, `rag_q4096_m20000_d128_k10`, `rag_lowq_q8_m131072_d128_k10`, `rag_lowq_q16_m131072_d128_k10`, `rag_lowq_q32_m131072_d128_k10`, `rag_lowq_q64_m131072_d128_k10`, `ksweep_q4096_m20000_d128_k1`, `ksweep_q4096_m20000_d128_k5`, `ksweep_q4096_m20000_d128_k8`, `round54_q4096_m16384_d128_k8`, `round54_q4096_m32768_d128_k8`.
- Candidate lifecycle latency diagnostics: init-once median `491.8068 ms`; first-signature compute median/p90 `40.5158/235.4393 ms`; hot compute median/p90 `0.2892/1.3681 ms`

#### Hot steady-state synchronized E2E speedup

| Validated shape scope | Min | Geomean | Median | P90 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| All 198 benchmarked shapes (diagnostic scope) | 0.0105x | 1.0556x | 1.5706x | 2.0658x | 4.1821x |

#### Modeled after-init amortized synchronized E2E speedup

| Public calls N | Min | Geomean | Median | P90 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0122x | 2.3699x | 2.6206x | 48.3276x | 507.2297x |
| 10 | 0.0231x | 2.1795x | 2.0424x | 33.0627x | 259.1853x |
| 100 | 0.0124x | 1.7163x | 1.4122x | 14.1772x | 59.3342x |
| 1000 | 0.0107x | 1.3001x | 1.4562x | 4.4227x | 9.9560x |

#### Modeled including-init amortized synchronized E2E speedup

| Public calls N | Min | Geomean | Median | P90 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0006x | 0.1422x | 0.0593x | 1.3762x | 4.6656x |
| 10 | 0.0059x | 0.1669x | 0.0656x | 1.3622x | 4.6575x |
| 100 | 0.0119x | 0.2731x | 0.1834x | 1.3362x | 4.5804x |
| 1000 | 0.0107x | 0.6010x | 0.5748x | 1.4267x | 4.0330x |

- All three tables report synchronized host E2E speedups as `baseline/candidate`. `Hot steady-state` measures a repeated public call at each lane's declared hot cache state; its per-shape values supply the official metric used by the separate publication-floor section.
- `After-init amortized(N) = (first_compute + (N-1) * hot_median) / N`; it excludes init.
- `Including-init amortized(N) = (init + first_compute + (N-1) * hot_median) / N`; it includes init. Each latency formula is evaluated separately for baseline and candidate, then reported as `baseline/candidate`. Both amortized scenarios are composed from measured components, not a directly timed N-call loop. A lane without explicit init uses `I=0`.
- Init scope: `once_per_validation_shard_process_device_operator`; composition: `runtime_init_only`; baseline has explicit init: `no`.
- Cache policy: `synchronize_and_clear_after_each_completed_shape`; resident multi-shape cache benchmarked: `no`; cold order: `deterministic_balanced_per_publication_contract_portfolio`; init order: `candidate_only_baseline_has_no_explicit_init`
- Lifecycle timing convention: all three lifecycle tables are synchronized host E2E. Init/first-call brackets are CUPTI timestamp host diagnostics; separately, the hot GPU-span diagnostic remains strict correlated CUPTI activity timing.

- Measured: `2026-07-07T10:05:23+00:00`
- Full summary: [`VALIDATION.json`](VALIDATION.json); per-shape results: [`BENCHMARK_RESULTS.json`](BENCHMARK_RESULTS.json)
