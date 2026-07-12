## Pre-publication GPU validation: PASS — declared 111-shape performance floor

- Hardware: `NVIDIA GB200` (`sm_100a`)
- Shapes: correctness `112/112`, CUPTI benchmark `112/112`; full generated suite `114` tests
- Validation shards: `4`; host wall time: correctness `6.34s`, benchmark `8.38s`
- Full all-shape `compute_speedup_vs_baseline` diagnostic vs `flashlib.flash_knn`: min `1.1398x`, geomean `2.4125x`, median `2.4369x`, p90 `4.0648x`, max `4.8001x`; `1/112` shapes are below the nominal `1.2000x` threshold (diagnostic, not hidden; see `VALIDATION.json`).
- Publication performance floor: `111` explicitly named shapes; min `1.2362x`, geomean `2.4288x`, median `2.4461x`, p90 `4.0673x`, max `4.8001x` (required minimum `1.2000x`).
- Publication floor labels: `flashml_correctness_b1_q256_m256_d128_k5`, `build_k_sweep_qm512_k1`, `build_k_sweep_qm512_k2`, `build_k_sweep_qm512_k4`, `build_k_sweep_qm512_k5`, `build_k_sweep_qm512_k6`, `build_k_sweep_qm512_k8`, `build_k_sweep_qm512_k10`, `build_qm1024_d128_k10`, `build_k_sweep_qm1024_k16`, `build_k_sweep_qm1024_k12`, `build_k_sweep_qm1024_k20`, `build_qm2048_d128_k8`, `build_qm1024_d128_k8`, `build_qm4096_d128_k8`, `build_qm2048_d128_k10`, `build_dim_sweep_b1_q1024_m1024_d64_k10`, `build_dim_sweep_b1_q2048_m2048_d64_k10`, `build_dim_sweep_b1_q4096_m4096_d64_k10`, `build_dim_sweep_b1_q1024_m1024_d96_k10`, `build_dim_sweep_b1_q2048_m2048_d192_k10`, `build_dim_sweep_b1_q2048_m2048_d256_k10`, `build_common_d256_b1_q1024_m1024_k10`, `build_common_d768_b1_q1024_m1024_k10`, `build_common_d1024_b1_q512_m512_k10`, `build_common_d4096_b1_q512_m512_k10`, `build_highd_b1_q1024_m1024_d320_k10`, `build_dtype_fp16_b1_q2048_m2048_d128_k10`, `build_batch_b2_q1024_m1024_d128_k10`, `build_k_sweep_qm2048_k11`, `build_k_sweep_qm2048_k12`, `build_k_sweep_qm2048_k13`, `build_k_sweep_qm2048_k20`, `build_k_sweep_qm2048_k24`, `build_k_sweep_qm2048_k28`, `build_tail_b1_q1536_m1536_d128_k10`, `build_tail_b1_q3072_m3072_d128_k20`, `build_medium_b1_q4096_m4096_d128_k10`, `build_k_sweep_qm4096_k12`, `build_k_sweep_qm4096_k13`, `build_k_sweep_qm4096_k20`, `build_k_sweep_qm4096_k24`, `build_k_sweep_qm4096_k28`, `build_largek_stress_qm4096_k32`, `build_k_sweep_qm4096_k30`, `build_over32_stress_qm2048_k48`, `build_over32_stress_qm2048_k64`, `build_over32_stress_qm4096_k48`, `build_large_b1_q8192_m8192_d128_k10`, `build_large_b1_q6144_m6144_d128_k10`, `build_large_b1_q8192_m8192_d128_k20`, `build_large_b1_q8192_m8192_d128_k32`, `build_verylarge_b1_q12288_m12288_d128_k10`, `rag_offline_b1_q4096_m100000_d128_k10`, `search_rect_b1_q1024_m8192_d128_k10`, `search_rect_b1_q1024_m32768_d64_k10`, `search_rect_highd_b1_q512_m12000_d320_k10`, `search_rect_common_d256_b1_q1024_m32768_k10`, `search_rect_common_d768_b1_q512_m8192_k10`, `search_rect_b1_q4096_m65536_d128_k20`, `search_rect_b1_q1536_m65536_d128_k20`, `search_rect_over32_b1_q2048_m65536_d128_k64`, `rag_online_b1_q1_m100000_d128_k10`, `rag_online_b1_q1_m65536_d128_k10`, `rag_online_irregular_b1_q1_m131071_d128_k10`, `rag_online_large_m_b1_q1_m250000_d128_k10`, `rag_online_irregular_b1_q1_m262143_d128_k10`, `rag_online_irregular_b1_q1_m524287_d128_k10`, `rag_stream_b1_q128_m100000_d128_k10`, `rag_offline_largek_b1_q4096_m100000_d128_k20`, `rag_offline_large_m_b1_q8192_m250000_d128_k20`, `rag_offline_large_m_over32_b1_q2048_m250000_d128_k64`, `rag_offline_batch_b1_q10000_m100000_d128_k10`, `rag_offline_b1_q10000_m50000_d128_k10`, `rag_microbatch_b1_q4_m100000_d128_k10`, `rag_microbatch_b1_q8_m100000_d128_k10`, `rag_microbatch_b1_q16_m100000_d128_k10`, `rag_microbatch_highd_b1_q16_m50000_d768_k10`, `rag_microbatch_common_d64_b1_q16_m50000_k10`, `rag_microbatch_common_d256_b1_q16_m50000_k10`, `rag_microbatch_common_d1024_b1_q8_m50000_k10`, `rag_microbatch_common_d4096_b1_q4_m32768_k10`, `rag_microbatch_b1_q32_m100000_d128_k10`, `rag_microbatch_largek_b1_q8_m100000_d128_k32`, `rag_microbatch_largek_b1_q16_m100000_d128_k32`, `rag_microbatch_largek_b1_q24_m100000_d128_k32`, `rag_microbatch_largek_b1_q16_m250000_d128_k32`, `rag_microbatch_largek_b1_q32_m100000_d128_k32`, `rag_microbatch_largek_b1_q16_m131071_d128_k32`, `rag_microbatch_b1_q64_m100000_d128_k10`, `rag_stream_largek_b1_q128_m100000_d128_k32`, `rag_stream_largek_b1_q128_m131071_d128_k32`, `rag_batch_b2_q256_m50000_d128_k10`, `rag_irregular_b1_q512_m131071_d128_k10`, `search_rect_b1_q2048_m32768_d128_k10`, `build_large_tail_b1_q6144_m6144_d128_k20`, `build_over32_stress_qm4096_k64`, `build_over64_stress_qm1024_k96`, `build_over64_stress_qm2048_k96`, `build_over64_stress_qm4096_k96`, `rag_online_common_d64_b1_q1_m262143_k10`, `rag_microbatch_common_d64_b1_q4_m100000_k10`, `rag_microbatch_common_d256_b1_q4_m100000_k10`, `rag_stream_common_d256_b1_q128_m100000_k10`, `rag_microbatch_common_d768_b1_q8_m100000_k10`, `rag_microbatch_common_d1024_b1_q4_m100000_k10`, `search_rect_common_d1024_b1_q256_m8192_k10`, `search_rect_common_d4096_b1_q128_m4096_k10`, `rag_microbatch_largek_common_d256_b1_q8_m100000_k32`, `rag_stream_largek_common_d256_b1_q128_m100000_k32`, `rag_microbatch_over32_d128_b1_q16_m100000_k48`.
- Candidate lifecycle latency diagnostics: init-once median `380.5765 ms`; first-signature compute median/p90 `6.3990/9.1849 ms`; hot compute median/p90 `0.1529/0.4773 ms`

#### Hot steady-state synchronized E2E speedup

| Validated shape scope | Min | Geomean | Median | P90 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| All 112 benchmarked shapes (diagnostic scope) | 1.1398x | 2.4125x | 2.4369x | 4.0648x | 4.8001x |

#### Modeled after-init amortized synchronized E2E speedup

| Public calls N | Min | Geomean | Median | P90 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0791x | 7.9179x | 8.4518x | 11.2692x | 30.1977x |
| 10 | 0.6569x | 6.7203x | 7.0840x | 9.8283x | 28.6100x |
| 100 | 1.5011x | 3.9494x | 3.9756x | 6.3274x | 19.4086x |
| 1000 | 1.2384x | 2.6508x | 2.6971x | 4.3696x | 7.6205x |

#### Modeled including-init amortized synchronized E2E speedup

| Public calls N | Min | Geomean | Median | P90 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0008x | 0.1426x | 0.1437x | 0.1782x | 1.2876x |
| 10 | 0.0080x | 0.1592x | 0.1540x | 0.1923x | 1.2967x |
| 100 | 0.0783x | 0.2665x | 0.2419x | 0.3962x | 1.3850x |
| 1000 | 0.5973x | 0.8763x | 0.8264x | 1.1917x | 2.0604x |

- All three tables report synchronized host E2E speedups as `baseline/candidate`. `Hot steady-state` measures a repeated public call at each lane's declared hot cache state; its per-shape values supply the official metric used by the separate publication-floor section.
- `After-init amortized(N) = (first_compute + (N-1) * hot_median) / N`; it excludes init.
- `Including-init amortized(N) = (init + first_compute + (N-1) * hot_median) / N`; it includes init. Each latency formula is evaluated separately for baseline and candidate, then reported as `baseline/candidate`. Both amortized scenarios are composed from measured components, not a directly timed N-call loop. A lane without explicit init uses `I=0`.
- Init scope: `once_per_validation_shard_process_device_operator`; composition: `runtime_init_only`; baseline has explicit init: `no`.
- Cache policy: `synchronize_and_clear_after_each_completed_shape`; resident multi-shape cache benchmarked: `no`; cold order: `deterministic_balanced_per_publication_contract_portfolio`; init order: `candidate_only_baseline_has_no_explicit_init`
- Lifecycle timing convention: all three lifecycle tables are synchronized host E2E. Init/first-call brackets are CUPTI timestamp host diagnostics; separately, the hot GPU-span diagnostic remains strict correlated CUPTI activity timing.

- Measured: `2026-07-12T01:41:19+00:00`
- Full summary: [`VALIDATION.json`](VALIDATION.json); per-shape results: [`BENCHMARK_RESULTS.json`](BENCHMARK_RESULTS.json)
