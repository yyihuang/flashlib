# Correctness And Performance Results

## Export Provenance

- Package: `flashlib_cake_kmeans`
- Source repository: `ssh://git@gitlab-master.nvidia.com:12051/cake/cake.git`
- Source commit: `159a6ddb4bfecffa63fc3bf177d319df52e77fb4`
- Generated at: `2026-07-12T01:45:17.952607+00:00`

## Latest Recorded Results

## Pre-publication GPU validation: PASS — declared 124-shape performance floor

- Hardware: `NVIDIA GB200` (`sm_100a`)
- Shapes: correctness `124/124`, CUPTI benchmark `124/124`; full generated suite `128` tests
- Validation shards: `4`; host wall time: correctness `13.97s`, benchmark `187.46s`
- `public_raw_e2e_speedup_vs_07cf_adapter` vs `triton_h200_07cf_raw_adapter_v1`: min `1.1381x`, geomean `2.2126x`, median `2.2243x`, p90 `2.8308x`, max `3.8152x` across all `124` floor-gated shapes (required minimum `1.0000x`)
- Candidate lifecycle latency diagnostics: init-once median `228.0409 ms`; first-signature compute median/p90 `2.8208/4.4589 ms`; hot compute median/p90 `0.0928/0.1582 ms`

#### Hot steady-state synchronized E2E speedup

| Validated shape scope | Min | Geomean | Median | P90 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| All 124 benchmarked shapes (diagnostic scope) | 1.1381x | 2.2126x | 2.2243x | 2.8308x | 3.8152x |

#### Modeled after-init amortized synchronized E2E speedup

| Public calls N | Min | Geomean | Median | P90 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.2171x | 3.1375x | 2.8154x | 3.5534x | 245.4951x |
| 10 | 1.3077x | 2.9846x | 2.6457x | 3.1795x | 198.3629x |
| 100 | 1.2351x | 2.6264x | 2.3528x | 3.0937x | 69.1645x |
| 1000 | 1.1489x | 2.3527x | 2.2455x | 3.1251x | 11.3386x |

#### Modeled including-init amortized synchronized E2E speedup

| Public calls N | Min | Geomean | Median | P90 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0164x | 1.6543x | 20.1529x | 129.7095x | 285.0459x |
| 10 | 0.0199x | 1.6807x | 17.8689x | 106.7921x | 246.1575x |
| 100 | 0.0533x | 1.7491x | 5.9877x | 43.4815x | 105.0620x |
| 1000 | 0.3311x | 1.8268x | 1.8014x | 8.7205x | 17.6637x |

- All three tables report synchronized host E2E speedups as `baseline/candidate`. `Hot steady-state` measures a repeated public call at each lane's declared hot cache state; its per-shape values supply the official metric used by the separate publication-floor section.
- `After-init amortized(N) = (first_compute + (N-1) * hot_median) / N`; it excludes init.
- `Including-init amortized(N) = (init + first_compute + (N-1) * hot_median) / N`; it includes init. Each latency formula is evaluated separately for baseline and candidate, then reported as `baseline/candidate`. Both amortized scenarios are composed from measured components, not a directly timed N-call loop. A lane without explicit init uses `I=0`.
- Init scope: `runtime_init_plus_standalone_shared_preprocess_support_once_per_validation_shard_process_device_operator`; composition: `runtime_init_plus_shared_preprocess_support_each_standalone_lane`; baseline has explicit init: `yes`.
- Cache policy: `synchronize_and_clear_after_each_completed_shape`; resident multi-shape cache benchmarked: `no`; cold order: `deterministic_balanced_per_publication_contract_portfolio`; init order: `alternate_candidate_baseline_first_by_validation_shard_parity_then_shared_support`
- Lifecycle timing convention: all three lifecycle tables are synchronized host E2E. Init/first-call brackets are CUPTI timestamp host diagnostics; separately, the hot GPU-span diagnostic remains strict correlated CUPTI activity timing.

- Measured: `2026-07-12T01:48:47+00:00`
- Full summary: [`VALIDATION.json`](VALIDATION.json); per-shape results: [`BENCHMARK_RESULTS.json`](BENCHMARK_RESULTS.json)

The machine-readable per-shape public lifecycle and CUPTI evidence is in [`BENCHMARK_RESULTS.json`](BENCHMARK_RESULTS.json).

## Result Table

| Test | Hardware | Command | Result |
| --- | --- | --- | --- |
| metadata unit tests | not required | `pytest tests/test_exported_kernels.py tests/test_benchmark_harness.py -q` | pending |
| NVRTC compile benchmark | CUDA host | `python benchmarks/benchmark_exported_kernels.py --arch sm_100a --json results/compile_benchmark.json` | pending |
| semantic correctness | target GPU | `pytest tests/test_correctness.py -q` | PASS |
| kernel performance | target GPU | `python benchmarks/benchmark.py --no-correctness` | PASS (declared publication floor) |

## Kernel Inventory

| Name | Symbol | Launch Mode | Threads | Shared Memory Bytes |
| --- | --- | --- | ---: | ---: |
| `dispatch_kernel_0000` | `kernel_flash_kmeans_assign_lowdim_pack_e50c_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0001` | `kernel_flash_kmeans_assign_lowdim_e50c_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0002` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `standard` | 256 | 0 |
| `dispatch_kernel_0003` | `kernel_flash_kmeans_assign_d160_pad192_pack_f9b2_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0004` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d192_single_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0005` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d192_splitd_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0006` | `kernel_flash_kmeans_assign_highd_splitd_6fcf_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0007` | `kernel_flash_kmeans_assign_highd_splitk_partial_blockn64_g2r4_b5a6_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0008` | `kernel_flash_kmeans_assign_highd_splitk_reduce_blockn64_g2r4_b5a6_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0009` | `kernel_flash_kmeans_assign_microdim_pack_6cd2_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0010` | `kernel_flash_kmeans_assign_microdim_6cd2_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0011` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0012` | `kernel_flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0013` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v15` | `standard` | 256 | 0 |
| `dispatch_kernel_0014` | `kernel_flash_kmeans_assign_highd_splitk_partial_blockn64_g1r4_streamdep_r63_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0015` | `kernel_flash_kmeans_assign_highd_splitk_reduce_blockn64_g1r4_streamdep_r63_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0016` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d160_pack_padded_b23d_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0017` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d256_single_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0018` | `kernel_flash_kmeans_assign_highd_paired_xreuse_dualtmem_producer_r47_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0019` | `kernel_flash_kmeans_assign_highd_paired_ownerreduce_r39_reduce1_unroll_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0020` | `kernel_flash_kmeans_assign_highd_paired_packedpartial_producer_7b3c_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0021` | `kernel_flash_kmeans_assign_highd_paired_packedpartial_reduce_r2_7b3c_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0022` | `kernel_flash_kmeans_assign_microdim_d16_pipeline4_08f9_v4` | `standard` | 256 | 0 |
| `dispatch_kernel_0023` | `kernel_flash_kmeans_assign_microdim_raw_tma_08f9_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0024` | `kernel_flash_kmeans_assign_d112_consumer_scoped_direct_mma_view_handoff_e5e1_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0025` | `kernel_flash_kmeans_assign_d224_tmem_abi_repair_d17c_v4` | `standard` | 256 | 0 |
| `dispatch_kernel_0026` | `kernel_flash_kmeans_assign_d288_exactd_a532_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0027` | `kernel_flash_kmeans_assign_d288_splitk_cta_0438_v1_partial` | `standard` | 256 | 0 |
| `dispatch_kernel_0028` | `kernel_flash_kmeans_assign_d288_splitk_cta_0438_v1_reduce` | `standard` | 256 | 0 |
| `dispatch_kernel_0029` | `kernel_flash_kmeans_assign_d480_splitk_partial_d32k256_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0030` | `kernel_flash_kmeans_assign_d480_splitk_reduce_d32k256_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0031` | `kernel_flash_kmeans_assign_highd_splitk_partial_8de8_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0032` | `kernel_flash_kmeans_assign_highd_splitk_reduce_8de8_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0033` | `kernel_flash_kmeans_assign_microdim_direct_9c0d_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0034` | `kernel_flash_kmeans_assign_d416_exactd_splitd_a4a579d1_v2` | `standard` | 256 | 0 |
| `dispatch_kernel_0035` | `kernel_flash_kmeans_assign_d112_k1024_owner_local_warp_mma_distinct_workfeed_c829_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0036` | `kernel_flash_kmeans_assign_d112_k512_two_owner_peer_only_mma_f826_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0037` | `kernel_flash_kmeans_assign_d112_shared_point_dual_issuer_tcgen05_6e1e_v1` | `standard` | 256 | 0 |
