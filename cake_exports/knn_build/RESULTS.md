# Correctness And Performance Results

## Export Provenance

- Package: `flashlib_cake_knn_build`
- Source repository: `ssh://git@gitlab-master.nvidia.com:12051/cake/cake.git`
- Source commit: `0e46454eee1c7aee5b5991fa2b0c6bc6a97293e4`
- Generated at: `2026-07-02T18:48:37.212801+00:00`

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
| `stage1_k32_unordered_ir` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered` | `cluster` | 192 | 50432 |
| `stage1_k12_ir` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k12split` | `cluster` | 192 | 50432 |
| `stage1_k16_ir` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k16split` | `cluster` | 192 | 50432 |
| `stage1_k20_ir` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k20split` | `cluster` | 192 | 50432 |
| `stage1_k25_ir` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k25split` | `cluster` | 192 | 50432 |
| `stage1_k30_ir` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k30split` | `cluster` | 192 | 50432 |
| `stage1_ir` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32split` | `cluster` | 192 | 50432 |
| `stage1_k20_unordered_ir` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k20unordered` | `cluster` | 192 | 50432 |
| `stage1_k30_unordered_ir` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k30unordered` | `cluster` | 192 | 50432 |
| `knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache` | `kernel_knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache` | `normal` | 32 | 0 |
| `merge_k12_ir` | `kernel_knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache_k12split` | `normal` | 32 | 0 |
| `merge_k16_ir` | `kernel_knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache_k16split` | `normal` | 32 | 0 |
| `merge_k20_ir` | `kernel_knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache_k20split` | `normal` | 32 | 0 |
| `merge_k25_ir` | `kernel_knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache_k25split` | `normal` | 32 | 0 |
| `merge_k30_ir` | `kernel_knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache_k30split` | `normal` | 32 | 0 |
| `merge_ir` | `kernel_knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache_k32split` | `normal` | 32 | 0 |
| `merge_k32_unordered_ir` | `kernel_knn_build_evolve_7bfc_k32_merge_s4_unordered` | `normal` | 32 | 0 |
| `merge_k20_unordered_ir` | `kernel_knn_build_evolve_7bfc_k32_merge_s4_unordered_k20unordered` | `normal` | 32 | 0 |
| `merge_k30_unordered_ir` | `kernel_knn_build_evolve_7bfc_k32_merge_s4_unordered_k30unordered` | `normal` | 32 | 0 |
| `merge_k30_s8_ir` | `kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache` | `normal` | 32 | 0 |
| `merge_k12_s8_ir` | `kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k12s8` | `normal` | 32 | 0 |
| `k5t64_stage1_ir` | `kernel_knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_threshold_k5_mintree` | `cluster` | 192 | 50432 |
| `k5t64_merge_ir` | `kernel_knn_build_evolve_7bfc_k5_merge_s4_tree_rowbase` | `normal` | 256 | 0 |
| `k5t64_merge_k10_s4_cache_ir` | `kernel_knn_build_evolve_7bfc_k10_merge_s4_rowbase_cache` | `normal` | 64 | 0 |
| `k5t64_merge_k10_s7_cache_ir` | `kernel_knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache` | `normal` | 256 | 0 |
| `k10t32_stage1_ir` | `kernel_knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_threshold_k5_mintree` | `cluster` | 192 | 50432 |
| `k10t32_merge_ir` | `kernel_knn_build_evolve_7bfc_k5_merge_s4_tree_rowbase` | `normal` | 256 | 0 |
| `k10t32_merge_k10_s4_cache_ir` | `kernel_knn_build_evolve_7bfc_k10_merge_s4_rowbase_cache` | `normal` | 32 | 0 |
| `k10t32_merge_k10_s7_cache_ir` | `kernel_knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache` | `normal` | 32 | 0 |
| `k10root_stage1` | `kernel_knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_maxtree` | `cluster` | 192 | 50432 |
| `knn_build_d64_build_aa88_v2_stage1_d64_split_ir` | `kernel_knn_build_dim_midk_73a9_d64_split_stage1` | `normal` | 192 | 25856 |
| `knn_build_d64_build_aa88_v2_merge_generic_ir` | `kernel_knn_build_evolve_7bfc_split_merge` | `normal` | 256 | 0 |
| `knn_build_d64_build_aa88_v2_merge_ir` | `kernel_knn_build_d64_build_aa88_k10_merge_s8_rowbase_cache` | `normal` | 32 | 0 |
| `knn_build_d64_build_aa88_v2_merge_k10_s4_ir` | `kernel_knn_build_d64_build_aa88_k10_merge_s8_rowbase_cache_s4` | `normal` | 32 | 0 |
| `knn_build_dim_midk_73a9_v1_stage1_d64_split_ir` | `kernel_knn_build_dim_midk_73a9_d64_split_stage1` | `normal` | 192 | 25856 |
| `knn_build_dim_midk_73a9_v1_merge_generic_ir` | `kernel_knn_build_evolve_7bfc_split_merge` | `normal` | 256 | 0 |
| `knn_build_dim_midk_bad5_k24k28_v1_stage1_k24_s8_ir` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_bad5k24s8` | `cluster` | 192 | 50432 |
| `knn_build_dim_midk_bad5_k24k28_v1_stage1_k28_s8_ir` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_bad5k28s8` | `cluster` | 192 | 50432 |
| `knn_build_dim_midk_bad5_k24k28_v1_stage1_k28_unordered_ir` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k30unordered_bad5k28unordered` | `cluster` | 192 | 50432 |
| `knn_build_dim_midk_bad5_k24k28_v1_merge_k24_s8_ir` | `kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_bad5k24s8` | `normal` | 32 | 0 |
| `knn_build_dim_midk_bad5_k24k28_v1_merge_k28_s8_ir` | `kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_bad5k28s8` | `normal` | 32 | 0 |
| `knn_build_dim_midk_bad5_k24k28_v1_merge_k28_unordered_ir` | `kernel_knn_build_evolve_7bfc_k32_merge_s4_unordered_k30unordered_bad5k28unordered` | `normal` | 32 | 0 |
| `knn_build_dim_midk_bad5_k64split8_v1_stage1_k64_s8_tailinf_ir` | `kernel_knn_build_k64_stage1_tailinf_k64over32tailinfsplitgrid` | `cluster` | 192 | 50432 |
| `knn_build_dim_midk_bad5_k64split8_v1_merge_k64_s8_warp_select_ir` | `kernel_knn_build_k64_merge_s8_unordered_warp_select_k64over32s8warpselect` | `normal` | 128 | 0 |
| `knn_build_dim_midk_bad5_midkcleanup_v1_stage1_k24_s8_ir` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_bad5midks8k24` | `cluster` | 192 | 50432 |
| `knn_build_dim_midk_bad5_midkcleanup_v1_stage1_k28_s8_ir` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_bad5midks8k28` | `cluster` | 192 | 50432 |
| `knn_build_dim_midk_bad5_midkcleanup_v1_merge_k24_s8_ir` | `kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_bad5midks8k24` | `normal` | 32 | 0 |
| `knn_build_dim_midk_bad5_midkcleanup_v1_merge_k28_s8_ir` | `kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_bad5midks8k28` | `normal` | 32 | 0 |
| `knn_build_dim_midk_df2f_v1_stage1_d256_split_ir` | `kernel_knn_build_dim_midk_df2f_d256_split_stage1` | `normal` | 192 | 99584 |
| `knn_build_dim_midk_df2f_v1_stage1_fp16_split_ir` | `kernel_knn_build_dim_midk_df2f_fp16_split_stage1` | `normal` | 192 | 50432 |
| `knn_build_dim_midk_df2f_v1_merge_generic_ir` | `kernel_knn_build_evolve_7bfc_split_merge` | `normal` | 256 | 0 |
| `knn_build_dim_midk_f8c3_q4096k64split_v1_stage1_k64_tailinf_ir` | `kernel_knn_build_k64_stage1_tailinf_k64over32tailinfsplitgrid` | `cluster` | 192 | 50432 |
| `knn_build_dim_midk_f8c3_q4096k64split_v1_merge_k64_s8_ir` | `kernel_knn_build_k64_merge_s8_unordered_warp_select_k64over32s8warpselect` | `normal` | 128 | 0 |
| `knn_build_dim_midk_f8c3_q4096k64split_v1_merge_k64_s12_ir` | `kernel_knn_build_k64_merge_sN_unordered_chunkprefill_k64over32s12chunkprefill` | `normal` | 32 | 0 |
| `knn_build_dim_midk_f8c3_q4096k64split_v1_merge_k64_s16_ir` | `kernel_knn_build_k64_merge_sN_unordered_chunkprefill_k64over32s16chunkprefill` | `normal` | 32 | 0 |
| `knn_build_dispatch_4fbf_7399_d15e_73a9_full55_v1` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32` | `cluster` | 192 | 50432 |
| `knn_build_dispatch_4fbf_7399_d15e_full55_bad5_v1` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32` | `cluster` | 192 | 50432 |
| `knn_build_dispatch_7399_d15e_df2f_full55_v1` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32` | `cluster` | 192 | 50432 |
| `knn_build_dispatch_7399_d15e_full55_v1` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32` | `cluster` | 192 | 50432 |
| `knn_build_dispatch_b6d4_d15e_fd02_v1` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32` | `cluster` | 192 | 50432 |
| `knn_build_dispatch_d64_fdd7_e3de_v1` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32` | `cluster` | 192 | 50432 |
| `knn_build_dispatch_e3de_9138_bcb3_4247_v1` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32` | `cluster` | 192 | 50432 |
| `knn_build_dispatch_rag_seed_portfolio_8700_v1` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32` | `cluster` | 192 | 50432 |
| `knn_build_dispatch_selected_portfolio_397b_v1` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32` | `cluster` | 192 | 50432 |
| `knn_build_dispatch_selected_portfolio_4a72_v1` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32` | `cluster` | 192 | 50432 |
| `knn_build_dispatch_selected_portfolio_e51c_v1` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32` | `cluster` | 192 | 50432 |
| `knn_build_dispatch_selected_portfolio_f16b_v1` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32` | `cluster` | 192 | 50432 |
| `knn_build_dispatch_selected_portfolio_f552_v1` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32` | `cluster` | 192 | 50432 |
| `knn_build_dispatch_selected_portfolio_f853_v1` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32` | `cluster` | 192 | 50432 |
| `knn_build_dispatch_selected_portfolio_f8c3_v1` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32` | `cluster` | 192 | 50432 |
| `knn_build_dispatch_split72_4e09_de1a_3dc7_v48` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32` | `cluster` | 192 | 50432 |
| `knn_build_large_square_k20k32_a989_v1` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered` | `cluster` | 192 | 50432 |
| `knn_build_large_square_k32_8a83_v1_merge_k32_s2_warp_select_ir` | `kernel_knn_build_large_square_k32_s2_warp_select` | `normal` | 128 | 0 |
| `knn_build_large_square_k32_8a83_v1_stage1_k32_split2_ir` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered` | `cluster` | 192 | 50432 |
| `knn_build_large_tail_frontier_6a73_v1` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k20split` | `cluster` | 192 | 50432 |
| `knn_build_lowk_f8c3_q512_q1024_v1_stage1_q512_lowk_ir` | `kernel_knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_maxtree` | `cluster` | 192 | 50432 |
| `knn_build_lowk_f8c3_q512_q1024_v1_merge_q512_generic_ir` | `kernel_knn_build_evolve_7bfc_split_merge` | `normal` | 256 | 0 |
| `knn_build_lowk_f8c3_q512_q1024_v1_stage1_q1024_k16_ir` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k16split` | `cluster` | 192 | 50432 |
| `knn_build_lowk_f8c3_q512_q1024_v1_merge_q1024_k16_s4_ir` | `kernel_knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache_k16split` | `normal` | 32 | 0 |
| `knn_build_lowk_f8c3_q512_q1024_v1_merge_q1024_k16_s8_ir` | `kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_f8c3lowk_k16s8` | `normal` | 32 | 0 |
| `knn_build_lowk_f8c3_q512_q1024_v1_merge_q1024_k16_s16_ir` | `kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_f8c3lowk_k16s16` | `normal` | 32 | 0 |
| `knn_build_over64_k96_a2f8_v1_stage1_k96_over64_ir` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k96over64` | `cluster` | 192 | 50432 |
| `knn_build_over64_k96_a2f8_v1_merge_k96_over64_ir` | `kernel_knn_build_evolve_7bfc_k32_merge_s4_unordered_k96over64` | `normal` | 32 | 0 |
| `knn_build_over64_k96_a2f8_v1_knn_build_k96_stage1_sort4_chunked` | `kernel_knn_build_k96_stage1_sort4_chunked` | `cluster` | 192 | 50432 |
| `knn_build_over64_k96_a2f8_v1_stage1_k96_sort4_chunked_over64_ir` | `kernel_knn_build_k96_stage1_sort4_chunked_k96over64sort4chunked` | `cluster` | 192 | 50432 |
| `knn_build_over64_k96_a2f8_v1_knn_build_k96_merge_s8_unordered_chunkprefill` | `kernel_knn_build_k96_merge_s8_unordered_chunkprefill` | `normal` | 32 | 0 |
| `knn_build_over64_k96_a2f8_v1_merge_k96_s8_chunkprefill_over64_ir` | `kernel_knn_build_k96_merge_s8_unordered_chunkprefill_k96over64s8chunkprefill` | `normal` | 32 | 0 |
| `knn_build_over64_k96_a989_v1_stage1_k96_over64_ir` | `kernel_knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k96over64` | `cluster` | 192 | 50432 |
| `knn_build_over64_k96_a989_v1_merge_k96_over64_ir` | `kernel_knn_build_evolve_7bfc_k32_merge_s4_unordered_k96over64` | `normal` | 32 | 0 |
| `knn_build_over64_k96_a989_v1_knn_build_k96_merge_s8_unordered_chunkprefill` | `kernel_knn_build_k96_merge_s8_unordered_chunkprefill` | `normal` | 32 | 0 |
| `knn_build_over64_k96_a989_v1_merge_k96_s8_chunkprefill_over64_ir` | `kernel_knn_build_k96_merge_s8_unordered_chunkprefill_k96over64s8chunkprefill` | `normal` | 32 | 0 |
| `knn_build_rag_frontier_4b5c_v1` | `kernel_knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k32s32_4b5c` | `normal` | 32 | 0 |
| `knn_build_rag_frontier_4fbf_v7_stage1_k32_tailinf_ir` | `kernel_knn_build_rag_frontier_4fbf_v7_stage1_k32_sort4earlystop_tailinf` | `cluster` | 192 | 50432 |
| `knn_build_rag_frontier_4fbf_v7_fused_merge_generalized_ir` | `kernel_knn_build_rag_frontier_4fbf_v7_k32_fused_group_final_merge` | `normal` | 32 | 8192 |
| `knn_build_rag_frontier_7399_v1` | `kernel_knn_build_rag_frontier_7399_k32_fused_group_final_merge` | `normal` | 32 | 2048 |
| `knn_build_rag_microbatch_4a72_v1_fused_merge_ir` | `kernel_knn_build_rag_microbatch_4a72_k10_fused_group_final_merge` | `normal` | 32 | 1024 |
| `knn_build_rag_microbatch_4a72_v1_ir` | `kernel_knn_build_rag_microbatch_4a72_k10_fused_group_final_merge_s72g8_4a72_v1` | `normal` | 32 | 1024 |
| `knn_build_rag_microbatch_4a72_v2_stage1_cta1_ir` | `kernel_knn_build_rag_microbatch_4a72_v2_stage1_k10_cta1_maxtree` | `normal` | 192 | 50432 |
| `knn_build_rag_microbatch_4a72_v2_fused_merge_ir` | `kernel_knn_build_rag_microbatch_4a72_v2_k10_fused_group_final_merge` | `normal` | 32 | 1024 |
| `knn_build_rag_microbatch_4a72_v2_ir` | `kernel_knn_build_rag_microbatch_4a72_v2_k10_fused_group_final_merge_s144g12_4a72_v2` | `normal` | 32 | 1024 |
| `knn_build_rag_microbatch_m64_d4f7_v1` | `kernel_knn_build_rag_microbatch_m64_d4f7_stage1` | `normal` | 512 | 91392 |
| `knn_build_ragonline_mbucket_aa88_q1m_v3_coop_merge_s72_k10_ir` | `kernel_knn_build_ragonline_mbucket_aa88_q1m_s72_k10_coop_merge` | `normal` | 128 | 512 |
| `knn_build_ragonline_mbucket_aa88_q1m_v3_coop_merge_s74_k10_ir` | `kernel_knn_build_ragonline_mbucket_aa88_q1m_s72_k10_coop_merge_s74_m250` | `normal` | 128 | 512 |
| `knn_build_ragonline_mbucket_aa88_q1m_v3_ir` | `kernel_knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_maxtree` | `cluster` | 192 | 50432 |
| `knn_build_rect_d64_cf49_v3_stage1_d64_split_ir` | `kernel_knn_build_dim_midk_73a9_d64_split_stage1` | `normal` | 192 | 25856 |
| `knn_build_rect_d64_cf49_v3_merge_generic_ir` | `kernel_knn_build_evolve_7bfc_split_merge` | `normal` | 256 | 0 |
| `knn_build_rect_d64_cf49_v3_merge_s16_cached_ir` | `kernel_knn_build_rect_d64_cf49_s16_cached_merge` | `normal` | 8 | 0 |
| `knn_build_rect_intermediate_frontier_6a73_4452_v2_merge_k10_s8_cache_ir` | `kernel_knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache_rect4452_s8` | `normal` | 32 | 0 |
| `knn_build_rect_intermediate_frontier_6a73_4452_v2_merge_k10_s12_cache_ir` | `kernel_knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache_rect4452_s12` | `normal` | 32 | 0 |
| `knn_build_rect_intermediate_frontier_6a73_4452_v2_merge_k10_s16_cache_ir` | `kernel_knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache_rect4452_s16` | `normal` | 32 | 0 |
| `knn_build_rect_intermediate_frontier_6a73_4452_v2_merge_k10_s24_cache_ir` | `kernel_knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache_rect4452_s24` | `normal` | 32 | 0 |
| `knn_build_rect_intermediate_frontier_6a73_4452_v2_merge_k10_s32_cache_ir` | `kernel_knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache_rect4452_s32` | `normal` | 32 | 0 |
| `knn_build_rect_intermediate_frontier_6a73_4452_v2_ir` | `kernel_knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_maxtree` | `cluster` | 192 | 50432 |
| `knn_build_rect_smallq_largem_ff59_d15e_v1_merge_k10_s8_cache_ir` | `kernel_knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache_rectd15e_s8` | `normal` | 32 | 0 |
| `knn_build_rect_smallq_largem_ff59_d15e_v1_merge_k10_s16_cache_ir` | `kernel_knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache_rectd15e_s16` | `normal` | 32 | 0 |
| `knn_build_rect_smallq_largem_ff59_d15e_v1_merge_k10_s32_cache_ir` | `kernel_knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache_rectd15e_s32` | `normal` | 32 | 0 |
| `knn_build_rect_smallq_largem_ff59_d15e_v1_ir` | `kernel_knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_maxtree` | `cluster` | 192 | 50432 |
