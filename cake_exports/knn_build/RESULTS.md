# Correctness And Performance Results

## Validated B200 result

- Correctness: `112/112`; route fidelity: `112/112`
- Public speedup: min `0.2314x`, geomean `0.7296x`, max `2.0338x`
- Prepared speedup: min `0.8233x`, geomean `2.0137x`, max `4.2444x`
- Prepared shapes above baseline: `111/112`
- Public/prepared GPU-span ratio: `2.7601x` geomean
- Result payload: [`BENCHMARK_RESULTS.json`](BENCHMARK_RESULTS.json)

## Export Provenance

- Package: `flashlib_cake_knn_build`
- Source repository: `https://gitlab-master.nvidia.com/cake/cake.git`
- Source commit: `90083f7caff1734b740c24083f944615d7fab15e`
- Generated at: `2026-07-04T09:10:13.861460+00:00`

## Latest Recorded Results

No semantic correctness or kernel throughput benchmark result was recorded by
the generic exporter at generation time.

The generated checks are:

```bash
pytest
python benchmarks/benchmark_exported_kernels.py --arch sm_100a --json results/compile_benchmark.json
python benchmarks/benchmark_shapes.py --json results/shape_benchmark.json
```

The shape runner requires a configured `benchmarks/workload.py` adapter. It
validates candidate output against the reference before running strict
CUPTI-backed, cold-L2 timing.

## Result Table

| Test | Hardware | Command | Result |
| --- | --- | --- | --- |
| metadata unit tests | not required | `pytest tests/test_exported_kernels.py tests/test_benchmark_harness.py -q` | pending |
| NVRTC compile benchmark | CUDA host | `python benchmarks/benchmark_exported_kernels.py --arch sm_100a --json results/compile_benchmark.json` | pending |
| semantic correctness | target GPU | `pytest tests/test_correctness.py -q` | pending |
| kernel performance | target GPU | `python benchmarks/benchmark.py --no-correctness` | pending |

## Kernel Inventory

| Name | Symbol | Launch Mode | Threads | Shared Memory Bytes |
| --- | --- | --- | ---: | ---: |
| `dispatch_kernel_0000` | `kernel_knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_threshold_k5` | `standard` | 256 | 0 |
| `dispatch_kernel_0001` | `kernel_knn_build_evolve_7bfc_k5_merge_s4_tree` | `standard` | 256 | 0 |
| `dispatch_kernel_0002` | `kernel_knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_maxtree` | `standard` | 256 | 0 |
| `dispatch_kernel_0003` | `kernel_knn_build_evolve_7bfc_split_merge` | `standard` | 256 | 0 |
| `dispatch_kernel_0004` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k8split` | `standard` | 256 | 0 |
| `dispatch_kernel_0005` | `kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k8s8` | `standard` | 256 | 0 |
| `dispatch_kernel_0006` | `kernel_knn_build_evolve_7bfc_k10_merge_s4_rowbase_cache` | `standard` | 256 | 0 |
| `dispatch_kernel_0007` | `kernel_knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache` | `standard` | 256 | 0 |
| `dispatch_kernel_0008` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k16split` | `standard` | 256 | 0 |
| `dispatch_kernel_0009` | `kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_f8c3lowk_k16s16` | `standard` | 256 | 0 |
| `dispatch_kernel_0010` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k12split` | `standard` | 256 | 0 |
| `dispatch_kernel_0011` | `kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k12s16` | `standard` | 256 | 0 |
| `dispatch_kernel_0012` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k20split` | `standard` | 256 | 0 |
| `dispatch_kernel_0013` | `kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k20s16` | `standard` | 256 | 0 |
| `dispatch_kernel_0014` | `kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_195e_q1024k8s16` | `standard` | 256 | 0 |
| `dispatch_kernel_0015` | `kernel_knn_build_q4096_k8_fd9b_stage1_unordered_exact_prefill` | `standard` | 256 | 0 |
| `dispatch_kernel_0016` | `kernel_knn_build_q4096_k8_fd9b_merge_s4_unordered_warp_select` | `standard` | 256 | 0 |
| `dispatch_kernel_0017` | `kernel_knn_build_dim_midk_73a9_d64_split_stage1` | `standard` | 256 | 0 |
| `dispatch_kernel_0018` | `kernel_knn_build_d64_build_aa88_k10_merge_s8_rowbase_cache` | `standard` | 256 | 0 |
| `dispatch_kernel_0019` | `kernel_knn_build_d64_q4096_c271_stage1_unordered_syncdrop` | `standard` | 256 | 0 |
| `dispatch_kernel_0020` | `kernel_knn_build_d64_build_aa88_k10_merge_s8_rowbase_cache_s4` | `standard` | 256 | 0 |
| `dispatch_kernel_0021` | `kernel_knn_build_non128_frontier_4be7_d96exact_stage1` | `standard` | 256 | 0 |
| `dispatch_kernel_0022` | `kernel_knn_build_non128_frontier_3d5a_k10_merge_s8_rowbase_cache` | `standard` | 256 | 0 |
| `dispatch_kernel_0023` | `kernel_knn_build_dim_midk_df2f_d256_split_stage1` | `standard` | 256 | 0 |
| `dispatch_kernel_0024` | `kernel_knn_build_non128_frontier_7231_pad_bf16_rows_d256` | `standard` | 256 | 0 |
| `dispatch_kernel_0025` | `kernel_knn_build_common_d_56f3_d256_q1024_k10_merge_rowbase_cache_s16` | `standard` | 256 | 0 |
| `dispatch_kernel_0026` | `kernel_knn_build_common_d768_build_eeff_m64split_stage1` | `standard` | 256 | 0 |
| `dispatch_kernel_0027` | `kernel_knn_build_non128_frontier_4be7_d768fused_merge_s16g8_4be7_d768fused_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0028` | `kernel_knn_build_non128_frontier_7231_stage1_d1024` | `standard` | 256 | 0 |
| `dispatch_kernel_0029` | `kernel_knn_build_common_d_56f3_k10_merge_rowbase_cache_s8` | `standard` | 256 | 0 |
| `dispatch_kernel_0030` | `kernel_knn_build_non128_frontier_7231_stage1_d4096` | `standard` | 256 | 0 |
| `dispatch_kernel_0031` | `kernel_knn_build_non128_frontier_8227_d320tail_stage1` | `standard` | 256 | 0 |
| `dispatch_kernel_0032` | `kernel_knn_build_dim_midk_df2f_fp16_split_stage1` | `standard` | 256 | 0 |
| `dispatch_kernel_0033` | `kernel_knn_build_fp16_d128_lowfloor_fd37_k10_s8_merge` | `standard` | 256 | 0 |
| `dispatch_kernel_0034` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_e080k11exact` | `standard` | 256 | 0 |
| `dispatch_kernel_0035` | `kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_e080k11s8exact` | `standard` | 256 | 0 |
| `dispatch_kernel_0036` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k12split` | `standard` | 256 | 0 |
| `dispatch_kernel_0037` | `kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k12s8` | `standard` | 256 | 0 |
| `dispatch_kernel_0038` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_e080k13exact` | `standard` | 256 | 0 |
| `dispatch_kernel_0039` | `kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_e080k13s8exact` | `standard` | 256 | 0 |
| `dispatch_kernel_0040` | `kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k20s8` | `standard` | 256 | 0 |
| `dispatch_kernel_0041` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_bad5midks8k24` | `standard` | 256 | 0 |
| `dispatch_kernel_0042` | `kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_bad5midks8k24` | `standard` | 256 | 0 |
| `dispatch_kernel_0043` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_bad5midks8k28` | `standard` | 256 | 0 |
| `dispatch_kernel_0044` | `kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_bad5midks8k28` | `standard` | 256 | 0 |
| `dispatch_kernel_0045` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k20split` | `standard` | 256 | 0 |
| `dispatch_kernel_0046` | `kernel_knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache_k20split` | `standard` | 256 | 0 |
| `dispatch_kernel_0047` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k12unordered` | `standard` | 256 | 0 |
| `dispatch_kernel_0048` | `kernel_knn_build_evolve_7bfc_k32_merge_s4_unordered_k12unordered` | `standard` | 256 | 0 |
| `dispatch_kernel_0049` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_2c1ck13unordered` | `standard` | 256 | 0 |
| `dispatch_kernel_0050` | `kernel_knn_build_evolve_7bfc_k32_merge_s4_unordered_2c1ck13unordered` | `standard` | 256 | 0 |
| `dispatch_kernel_0051` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k20unordered` | `standard` | 256 | 0 |
| `dispatch_kernel_0052` | `kernel_knn_build_evolve_7bfc_k32_merge_s4_unordered_k20unordered` | `standard` | 256 | 0 |
| `dispatch_kernel_0053` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k30unordered_1074k24unordered` | `standard` | 256 | 0 |
| `dispatch_kernel_0054` | `kernel_knn_build_1074_k24_q4096_merge_s4_unordered_warp_select` | `standard` | 256 | 0 |
| `dispatch_kernel_0055` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k30unordered_bad5k28unordered` | `standard` | 256 | 0 |
| `dispatch_kernel_0056` | `kernel_knn_build_evolve_7bfc_k32_merge_s4_unordered_k30unordered_bad5k28unordered` | `standard` | 256 | 0 |
| `dispatch_kernel_0057` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered` | `standard` | 256 | 0 |
| `dispatch_kernel_0058` | `kernel_knn_build_evolve_7bfc_k32_merge_s4_unordered_warp_select` | `standard` | 256 | 0 |
| `dispatch_kernel_0059` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k30unordered` | `standard` | 256 | 0 |
| `dispatch_kernel_0060` | `kernel_knn_build_k30_q4096_6998_merge_s4_unordered_warp_select` | `standard` | 256 | 0 |
| `dispatch_kernel_0061` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32` | `standard` | 256 | 0 |
| `dispatch_kernel_0062` | `kernel_knn_build_k48_merge_s4_unordered_warp_select` | `standard` | 256 | 0 |
| `dispatch_kernel_0063` | `kernel_knn_build_k64_stage1_tailinf_k64over32tailinfsplitgrid` | `standard` | 256 | 0 |
| `dispatch_kernel_0064` | `kernel_knn_build_k64_merge_s8_unordered_warp_select_k64over32s8warpselect` | `standard` | 256 | 0 |
| `dispatch_kernel_0065` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k20unordered` | `standard` | 256 | 0 |
| `dispatch_kernel_0066` | `kernel_knn_build_k20_mergeown_08ec_warp8_select_s2warp8` | `standard` | 256 | 0 |
| `dispatch_kernel_0067` | `kernel_knn_build_large_square_k32_stage1_chunkworst` | `standard` | 256 | 0 |
| `dispatch_kernel_0068` | `kernel_knn_build_large_square_k32_s2_warp8_merge` | `standard` | 256 | 0 |
| `dispatch_kernel_0069` | `kernel_knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache_rectd15e_s16` | `standard` | 256 | 0 |
| `dispatch_kernel_0070` | `kernel_knn_build_rect_d64_23be_s16_cached_merge` | `standard` | 256 | 0 |
| `dispatch_kernel_0071` | `kernel_knn_build_rect_d64_23be_unordered_stage1` | `standard` | 256 | 0 |
| `dispatch_kernel_0072` | `kernel_knn_build_non128_frontier_7231_stage1_d256` | `standard` | 256 | 0 |
| `dispatch_kernel_0073` | `kernel_knn_build_non128_frontier_7231_stage1_d768` | `standard` | 256 | 0 |
| `dispatch_kernel_0074` | `kernel_knn_build_non128_frontier_4be7_d768fused_merge_s32g8_4be7_d768fused_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0075` | `kernel_knn_build_evolve_7bfc_k20_merge_s4_unordered_warp_select` | `standard` | 256 | 0 |
| `dispatch_kernel_0076` | `kernel_knn_build_rect_d128_k20_q1536_warp4_merge` | `standard` | 256 | 0 |
| `dispatch_kernel_0077` | `kernel_knn_build_k64_stage1_tailinf_k64over32tailinfsplitgrid` | `standard` | 256 | 0 |
| `dispatch_kernel_0078` | `kernel_knn_build_k64_merge_s8_unordered_warp_select_k64over32s8warpselect` | `standard` | 256 | 0 |
| `dispatch_kernel_0079` | `kernel_knn_build_ragonline_mbucket_4fc7_q1m262_v2_stage1_q1_k10_m64_halfrow` | `standard` | 256 | 0 |
| `dispatch_kernel_0080` | `kernel_knn_build_rag_microbatch_4a72_k10_fused_group_final_merge_s144g12_4a72_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0081` | `kernel_knn_build_ragonline_mbucket_ea43_q1m524_n128_stage1` | `standard` | 256 | 0 |
| `dispatch_kernel_0082` | `kernel_knn_build_rag_microbatch_4a72_k10_fused_group_final_merge_s147g7_4a72_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0083` | `kernel_knn_build_rag_stream_k10_q128_1bed_rowld_stage1` | `standard` | 256 | 0 |
| `dispatch_kernel_0084` | `kernel_knn_build_d128_rag_q128_k10_s74_warp_merge_rowld_s74_1bed_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0085` | `kernel_knn_build_k20_large_lowfanout_s2_warp_select` | `standard` | 256 | 0 |
| `dispatch_kernel_0086` | `kernel_knn_build_rag_microbatch_4a72_v2_stage1_k10_cta1_maxtree` | `standard` | 256 | 0 |
| `dispatch_kernel_0087` | `kernel_knn_build_rag_microbatch_4a72_v2_k10_fused_group_final_merge_s144g12_4a72_v2` | `standard` | 256 | 0 |
| `dispatch_kernel_0088` | `kernel_knn_build_rag_microbatch_m64_d4f7_stage1` | `standard` | 256 | 0 |
| `dispatch_kernel_0089` | `kernel_knn_build_rag_microbatch_4a72_k10_fused_group_final_merge_s136g8_4a72_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0090` | `kernel_knn_build_non128_frontier_7ee5_m64rag_stage1` | `standard` | 256 | 0 |
| `dispatch_kernel_0091` | `kernel_knn_build_non128_frontier_4be7_d768fused_merge_s72g8_4be7_d768fused_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0092` | `kernel_knn_build_common_d_1438_rag_d64_m128_stage1` | `standard` | 256 | 0 |
| `dispatch_kernel_0093` | `kernel_knn_build_non128_frontier_4be7_d768fused_merge_s136g8_4be7_d768fused_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0094` | `kernel_knn_build_non128_frontier_7ee5_m64rag_stage1_d256_5e7f_rag_d64d256_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0095` | `kernel_knn_build_non128_frontier_4be7_d768fused_merge_s144g8_4be7_d768fused_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0096` | `kernel_knn_build_non128_frontier_7ee5_m64rag_stage1_d1024_5e7f_highd_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0097` | `kernel_knn_build_non128_frontier_4be7_d768fused_merge_s144g12_4be7_d768fused_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0098` | `kernel_knn_build_non128_frontier_7ee5_m64rag_stage1_d4096_5e7f_highd_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0099` | `kernel_knn_build_non128_frontier_4be7_d768fused_merge_s128g8_4be7_d768fused_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0100` | `kernel_knn_build_rag_microbatch_4a72_k10_fused_group_final_merge_s128g8_4a72_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0101` | `kernel_knn_build_rag_microbucket_k32q8half_0077_v1_stage1_q8_k32_m64_halfrow_q8half_0077_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0102` | `kernel_knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge_k32s144_0077_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0103` | `kernel_knn_build_rag_microbucket_k32_q16irreg2warp_a444_v2_stage1_q16_rowld1_2warp_q16dual2warp_56ed_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0104` | `kernel_knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge_k32s144r4_56ed_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0105` | `kernel_knn_build_rag_microbucket_k32rowld1warp_0077_v1_stage1_q32_k32_m64_rowld2_q24rowld2_24dc_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0106` | `kernel_knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge_k32q24s144r4_24dc_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0107` | `kernel_knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge_k32s288r4_56ed_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0108` | `kernel_knn_build_rag_microbucket_k32_q32rowld2exact_s141_72d1_v1_stage1_q32rowld2exact_f653_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0109` | `kernel_knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge_k32q32exact_s141r4_f653_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0110` | `kernel_knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge_k32s148r4_56ed_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0111` | `kernel_knn_build_rag_stream_k32_q128m100000_staticn128_664a_stage1` | `standard` | 256 | 0 |
| `dispatch_kernel_0112` | `kernel_knn_build_rag_frontier_7399_k32_fused_group_final_merge_k32s72g8_4fbf_v6` | `standard` | 256 | 0 |
| `dispatch_kernel_0113` | `kernel_knn_build_rag_microbucket_q32rowld_e5db_v1_stage1_q32_k32_m64` | `standard` | 256 | 0 |
| `dispatch_kernel_0114` | `kernel_knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge_k32s72_0077_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0115` | `kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k10s72_4e09` | `standard` | 256 | 0 |
| `dispatch_kernel_0116` | `kernel_knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache_rect4452_s8` | `standard` | 256 | 0 |
| `dispatch_kernel_0117` | `kernel_knn_build_k64_stage1_tailinf_k64over32tailinfsplitgrid` | `standard` | 256 | 0 |
| `dispatch_kernel_0118` | `kernel_knn_build_k64_merge_s8_unordered_warp_select_k64over32s8warpselect` | `standard` | 256 | 0 |
| `dispatch_kernel_0119` | `kernel_knn_build_k96_stage1_exact_prefill_q1024_k96over64exactprefillq1024_e5db` | `standard` | 256 | 0 |
| `dispatch_kernel_0120` | `kernel_knn_build_k96_merge_s8_unordered_chunkprefill_k96over64s2chunkprefill_f9d1` | `standard` | 256 | 0 |
| `dispatch_kernel_0121` | `kernel_knn_build_k96_merge_s8_unordered_chunkprefill_k96over64s4chunkprefill_f9d1` | `standard` | 256 | 0 |
| `dispatch_kernel_0122` | `kernel_knn_build_v12_d64_tail_017a_stage1` | `standard` | 256 | 0 |
| `dispatch_kernel_0123` | `kernel_knn_build_common_d768_build_eeff_m64split_stage1_d256_q128_k10_59fe_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0124` | `kernel_knn_build_non128_frontier_7ee5_m64rag_stage1_d768_5e7f_highd_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0125` | `kernel_knn_build_common_d768_build_eeff_m64split_stage1_d1024_be66_search_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0126` | `kernel_knn_build_non128_frontier_4be7_d768fused_merge_s64g8_4be7_d768fused_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0127` | `kernel_knn_build_common_d768_build_eeff_m64split_stage1_d4096_be66_search_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0128` | `kernel_knn_build_v12_d256_k32_tail_59fe_v1_stage1_rowld` | `standard` | 256 | 0 |
| `dispatch_kernel_0129` | `kernel_knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge_k32s64_0077_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0130` | `kernel_knn_build_v12_d128_q16_k48_dd2b_v1_stage1` | `standard` | 256 | 0 |
| `dispatch_kernel_0131` | `kernel_knn_build_rag_microbucket_k32warpmerge_0077_v1_warp_row_merge_k48s144r4_dd2b_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0132` | `kernel_knn_build_rag_stream_k10_s72_warp_row_merge_34da` | `standard` | 256 | 0 |
| `dispatch_kernel_0133` | `kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k10s64_3d97` | `standard` | 256 | 0 |
| `dispatch_kernel_0134` | `kernel_knn_build_evolve_7bfc_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0135` | `kernel_knn_build_evolve_7bfc_split_stage1` | `standard` | 256 | 0 |
| `dispatch_kernel_0136` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1` | `standard` | 256 | 0 |
| `dispatch_kernel_0137` | `kernel_knn_build_evolve_7bfc_k10_merge_s4` | `standard` | 256 | 0 |
| `dispatch_kernel_0138` | `kernel_knn_build_evolve_7bfc_k10_merge_s7` | `standard` | 256 | 0 |
| `dispatch_kernel_0139` | `kernel_knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_threshold_k5_mintree` | `standard` | 256 | 0 |
| `dispatch_kernel_0140` | `kernel_knn_build_evolve_7bfc_k5_merge_s4_tree_rowbase` | `standard` | 256 | 0 |
| `dispatch_kernel_0141` | `kernel_knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache` | `standard` | 256 | 0 |
| `dispatch_kernel_0142` | `kernel_knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_maxtree` | `standard` | 256 | 0 |
| `dispatch_kernel_0143` | `kernel_knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_threshold_k5_mintree` | `standard` | 256 | 0 |
| `dispatch_kernel_0144` | `kernel_knn_build_evolve_7bfc_k5_merge_s4_tree_rowbase` | `standard` | 256 | 0 |
| `dispatch_kernel_0145` | `kernel_knn_build_evolve_7bfc_k10_merge_s4_rowbase_cache` | `standard` | 256 | 0 |
| `dispatch_kernel_0146` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered` | `standard` | 256 | 0 |
| `dispatch_kernel_0147` | `kernel_knn_build_evolve_7bfc_k32_merge_s4_unordered` | `standard` | 256 | 0 |
| `dispatch_kernel_0148` | `kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache` | `standard` | 256 | 0 |
| `dispatch_kernel_0149` | `kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k12s8` | `standard` | 256 | 0 |
| `dispatch_kernel_0150` | `kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k20s8` | `standard` | 256 | 0 |
| `dispatch_kernel_0151` | `kernel_knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache_k8s7` | `standard` | 256 | 0 |
| `dispatch_kernel_0152` | `kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k20s16` | `standard` | 256 | 0 |
| `dispatch_kernel_0153` | `kernel_knn_build_evolve_7bfc_fp16_d128_base` | `standard` | 256 | 0 |
| `dispatch_kernel_0154` | `kernel_knn_build_evolve_7bfc_d256_twomma_base` | `standard` | 256 | 0 |
| `dispatch_kernel_0155` | `kernel_knn_build_evolve_7bfc_d64_tcgen05_base` | `standard` | 256 | 0 |
| `dispatch_kernel_0156` | `kernel_knn_build_evolve_7bfc_k20_merge_s4_unordered_warp_select_splitmajor` | `standard` | 256 | 0 |
| `dispatch_kernel_0157` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k20unordered` | `standard` | 256 | 0 |
| `dispatch_kernel_0158` | `kernel_knn_build_k20_mergeown_08ec_s4_rowbase_lane` | `standard` | 256 | 0 |
| `dispatch_kernel_0159` | `kernel_knn_build_k20_large_rect_s3_warp_select` | `standard` | 256 | 0 |
| `dispatch_kernel_0160` | `kernel_knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_maxtree` | `standard` | 256 | 0 |
| `dispatch_kernel_0161` | `kernel_knn_build_rag_frontier_b6d4_stage1_k32_sort4earlystop` | `standard` | 256 | 0 |
| `dispatch_kernel_0162` | `kernel_knn_build_rag_frontier_7399_k32_fused_group_final_merge` | `standard` | 256 | 0 |
| `dispatch_kernel_0163` | `kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k32s32_4b5c` | `standard` | 256 | 0 |
| `dispatch_kernel_0164` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k96over64` | `standard` | 256 | 0 |
| `dispatch_kernel_0165` | `kernel_knn_build_k96_merge_s8_unordered_chunkprefill_k96over64s8chunkprefill` | `standard` | 256 | 0 |
| `dispatch_kernel_0166` | `kernel_knn_build_rag_frontier_b6d4_stage1_k32_chunked` | `standard` | 256 | 0 |
| `dispatch_kernel_0167` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered` | `standard` | 256 | 0 |
| `dispatch_kernel_0168` | `kernel_knn_build_evolve_7bfc_k32_merge_s4_unordered` | `standard` | 256 | 0 |
| `dispatch_kernel_0169` | `kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache` | `standard` | 256 | 0 |
| `dispatch_kernel_0170` | `kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k12s8` | `standard` | 256 | 0 |
| `dispatch_kernel_0171` | `kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k20s8` | `standard` | 256 | 0 |
| `dispatch_kernel_0172` | `kernel_knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache_k8s7` | `standard` | 256 | 0 |
| `dispatch_kernel_0173` | `kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k8s8` | `standard` | 256 | 0 |
| `dispatch_kernel_0174` | `kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k20s16` | `standard` | 256 | 0 |
| `dispatch_kernel_0175` | `kernel_knn_build_rag_frontier_4fbf_v7_stage1_k32_sort4earlystop_tailinf` | `standard` | 256 | 0 |
| `dispatch_kernel_0176` | `kernel_knn_build_rag_frontier_4fbf_stage1_k32_sort4earlystop_tailinf` | `standard` | 256 | 0 |
| `dispatch_kernel_0177` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_bad5k24s8` | `standard` | 256 | 0 |
| `dispatch_kernel_0178` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_bad5k28s8` | `standard` | 256 | 0 |
| `dispatch_kernel_0179` | `kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_bad5k24s8` | `standard` | 256 | 0 |
| `dispatch_kernel_0180` | `kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_bad5k28s8` | `standard` | 256 | 0 |
| `dispatch_kernel_0181` | `kernel_knn_build_large_square_k32_s2_warp_select` | `standard` | 256 | 0 |
| `dispatch_kernel_0182` | `kernel_knn_build_rect_d64_cf49_s16_cached_merge` | `standard` | 256 | 0 |
| `dispatch_kernel_0183` | `kernel_knn_build_ragonline_mbucket_aa88_q1m_s72_k10_coop_merge` | `standard` | 256 | 0 |
| `dispatch_kernel_0184` | `kernel_knn_build_ragonline_mbucket_aa88_q1m_s72_k10_coop_merge_s74_m250` | `standard` | 256 | 0 |
| `dispatch_kernel_0185` | `kernel_knn_build_rag_microbucket_5093_v1_stage1_k32_tailinf_cta1_compactwarp` | `standard` | 256 | 0 |
| `dispatch_kernel_0186` | `kernel_knn_build_rag_microbucket_3505_v3_stage1_k32_tailinf_cta1` | `standard` | 256 | 0 |
| `dispatch_kernel_0187` | `kernel_knn_build_rect_d64_cf49_s16_cached_merge` | `standard` | 256 | 0 |
| `dispatch_kernel_0188` | `kernel_knn_build_k96_stage1_sort4_chunked_k96over64sort4chunked` | `standard` | 256 | 0 |
| `dispatch_kernel_0189` | `kernel_knn_build_k96_merge_s8_unordered_chunkprefill_k96over64s8chunkprefill` | `standard` | 256 | 0 |
| `dispatch_kernel_0190` | `kernel_knn_build_rag_microbucket_3505_v9_stage1_q8_k32_m64` | `standard` | 256 | 0 |
| `dispatch_kernel_0191` | `kernel_knn_build_d128_rag_q128_k10_s74_warp_merge` | `standard` | 256 | 0 |
| `dispatch_kernel_0192` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k12split` | `standard` | 256 | 0 |
| `dispatch_kernel_0193` | `kernel_knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache_e080k11s4exact` | `standard` | 256 | 0 |
| `dispatch_kernel_0194` | `kernel_knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache_e080k13s4exact` | `standard` | 256 | 0 |
| `dispatch_kernel_0195` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered` | `standard` | 256 | 0 |
| `dispatch_kernel_0196` | `kernel_knn_build_evolve_7bfc_k32_merge_s4_unordered` | `standard` | 256 | 0 |
| `dispatch_kernel_0197` | `kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache` | `standard` | 256 | 0 |
| `dispatch_kernel_0198` | `kernel_knn_build_rag_microbucket_k32rowld1warp_0077_v1_stage1_q16_k32_m64_rowld1_q16rowld1_0077_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0199` | `kernel_knn_build_rag_microbucket_k32rowld1warp_0077_v1_stage1_q32_k32_m64_rowld2_q32rowld2_0077_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0200` | `kernel_knn_build_k96_merge_s8_unordered_chunkprefill_k96over64s2chunkprefill_f9d1` | `standard` | 256 | 0 |
| `dispatch_kernel_0201` | `kernel_knn_build_k96_stage1_sort4_prefill_q1024_k96over64sort4prefillq1024_8c56` | `standard` | 256 | 0 |
| `dispatch_kernel_0202` | `kernel_knn_build_non128_frontier_8199_d384_stage1` | `standard` | 256 | 0 |
| `dispatch_kernel_0203` | `kernel_knn_build_rag_microbucket_q32_k31_c3d2_v1_stage1_q32k31_c3d2_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0204` | `kernel_knn_build_rag_microbucket_k32_f590_q32exact_v1_stage1_q32exact_f590_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0205` | `kernel_knn_build_rag_microbucket_k12_2f22_q48exact_v1_stage1` | `standard` | 256 | 0 |
| `dispatch_kernel_0206` | `kernel_knn_build_rag_microbucket_k32_q32rowld2uneven_f653_v1_stage1_q32rowld2uneven_f653_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0207` | `kernel_knn_build_rag_microbucket_k32rowld1warp_0077_v1_stage1_q32_k32_m64_rowld2_q32rowld2_f653_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0208` | `kernel_knn_build_rag_microbucket_k32_0cb5_q31tail_v2_stage1_q31exact_0cb5_v2` | `standard` | 256 | 0 |
| `dispatch_kernel_0209` | `kernel_knn_build_rag_microbucket_k32_q16irreg2warp_a444_v2_stage1_q16_rowld1_2warp_q16irreg2warp_a444_v2` | `standard` | 256 | 0 |
| `dispatch_kernel_0210` | `kernel_knn_build_rag_microbucket_q32rowld_e5db_v1_stage1_q32_k32_m64_q128rowld_60fb_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0211` | `kernel_knn_build_common_d_5e7f_rag_d64_m64_stage1` | `standard` | 256 | 0 |
| `dispatch_kernel_0212` | `kernel_knn_build_rag_microbucket_3505_v2_stage1_k32_tailinf_cta1` | `standard` | 256 | 0 |
| `dispatch_kernel_0213` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered` | `standard` | 256 | 0 |
| `dispatch_kernel_0214` | `kernel_knn_build_evolve_7bfc_k32_merge_s4_unordered` | `standard` | 256 | 0 |
| `dispatch_kernel_0215` | `kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache` | `standard` | 256 | 0 |
| `dispatch_kernel_0216` | `kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k12s8` | `standard` | 256 | 0 |
| `dispatch_kernel_0217` | `kernel_knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache_k8s7` | `standard` | 256 | 0 |
| `dispatch_kernel_0218` | `kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k8s8` | `standard` | 256 | 0 |
| `dispatch_kernel_0219` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_fd9b_k8unordered` | `standard` | 256 | 0 |
| `dispatch_kernel_0220` | `kernel_knn_build_d64_q4096_c271_twostage_group_reduce` | `standard` | 256 | 0 |
| `dispatch_kernel_0221` | `kernel_knn_build_d64_build_aa88_k10_merge_s8_rowbase_cache_c271_s5` | `standard` | 256 | 0 |
| `dispatch_kernel_0222` | `kernel_knn_build_d64_build_aa88_k10_merge_s8_rowbase_cache_c271_s6` | `standard` | 256 | 0 |
| `dispatch_kernel_0223` | `kernel_knn_build_d64_q4096_c271_stage1_syncdrop` | `standard` | 256 | 0 |
| `dispatch_kernel_0224` | `kernel_knn_build_d64_build_aa88_k10_merge_s8_rowbase_cache_c271_s5` | `standard` | 256 | 0 |
| `dispatch_kernel_0225` | `kernel_knn_build_d64_build_aa88_k10_merge_s8_rowbase_cache_c271_s6` | `standard` | 256 | 0 |
| `dispatch_kernel_0226` | `kernel_knn_build_common_d_generic_direct_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0227` | `kernel_knn_build_common_d_generic_direct_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0228` | `kernel_knn_build_common_d_5e7f_rag_d64_repair_stage1` | `standard` | 256 | 0 |
| `dispatch_kernel_0229` | `kernel_knn_build_rag_microbucket_k32_q32rowld2exact_f653_v1_stage1_q32rowld2exact_f653_v1` | `standard` | 256 | 0 |
| `dispatch_kernel_0230` | `kernel_knn_build_k96_merge_s2_unordered_warp_select` | `standard` | 256 | 0 |
| `dispatch_kernel_0231` | `kernel_knn_build_q1m524_workfeed_s147_g21_register_merge` | `standard` | 256 | 0 |
