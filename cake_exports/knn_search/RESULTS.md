# Correctness And Performance Results

## Export Provenance

- Package: `flashlib_cake_knn_search`
- Source repository: `ssh://git@gitlab-master.nvidia.com:12051/cake/cake.git`
- Source commit: `0e46454eee1c7aee5b5991fa2b0c6bc6a97293e4`
- Generated at: `2026-07-02T18:49:24.855961+00:00`

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
| `search_knn_search_warp_direct_v1` | `kernel_knn_search_warp_direct_v1` | `normal` | 256 | 640 |
| `search_knn_search_warp_split_partial_v1` | `kernel_knn_search_warp_split_partial_v1` | `normal` | 256 | 0 |
| `search_knn_search_warp_split_merge_v1` | `kernel_knn_search_warp_split_merge_v1` | `normal` | 256 | 20480 |
| `search_q1_knn_search_q1_tile_reduce_partial_v1` | `kernel_knn_search_q1_tile_reduce_partial_v1` | `normal` | 256 | 5120 |
| `search_q1_knn_search_q1_tile_reduce_merge_v1` | `kernel_knn_search_q1_tile_reduce_merge_v1` | `normal` | 256 | 640 |
| `knn_search_base_ir` | `kernel_knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1` | `normal` | 512 | 96000 |
| `knn_search_b2_q128_ir` | `kernel_knn_search_mma_split_partial_v1` | `normal` | 640 | 108800 |
| `knn_search_blockm640_ir` | `kernel_knn_search_lowq_tile_reduce_partial_0614_r10_e864_blockm640_v1` | `normal` | 256 | 5120 |
| `knn_search_blockm640_merge_ir` | `kernel_knn_search_lowq_tile_reduce_merge_0614_r10_e864_blockm640_v1` | `normal` | 256 | 640 |
| `knn_search_b2_q128_merge_ir` | `kernel_knn_search_mma_split_merge_stream_v1` | `normal` | 32 | 0 |
| `knn_search_blockm896_ir` | `kernel_knn_search_lowq_tile_reduce_partial_0614_r11_e864_blockm896_v1` | `normal` | 256 | 5120 |
| `knn_search_blockm896_merge_ir` | `kernel_knn_search_lowq_tile_reduce_merge_0614_r11_e864_blockm896_v1` | `normal` | 256 | 640 |
| `knn_search_q3_partial_ir` | `kernel_knn_search_lowq_tile_reduce_partial_m131072_exact_0617_cc76_v1` | `normal` | 256 | 5120 |
| `knn_search_q3_merge_ir` | `kernel_knn_search_lowq_tile_reduce_merge_m131072_exact_0617_cc76_v1` | `normal` | 128 | 384 |
| `knn_search_scalar_capacity_ir` | `kernel_knn_search_scalar_capacity_partial_v1` | `normal` | 256 | 0 |
| `knn_search_k1_merge8_partial_ir` | `kernel_knn_search_q4096_lowk_k1partial_onestage_0614_r2_3ff5_v1` | `normal` | 640 | 108800 |
| `knn_search_k1_merge8_merge_ir` | `kernel_knn_search_k1_top1_margin_0614_r93_merge8_v1` | `normal` | 128 | 0 |
| `knn_search_q1_m262144_ir` | `kernel_knn_search_q1_irregular_m_tail_partial_v1` | `normal` | 256 | 5120 |
| `knn_search_q1_m262144_merge_ir` | `kernel_knn_search_q1_flashdecode_merge128_0614_r92_v1` | `normal` | 256 | 640 |
| `knn_search_b2_k64_partial_ir` | `kernel_knn_search_80a5_b2_q128m65536_k64_twotile_partial_74f4_v1` | `normal` | 512 | 165120 |
| `knn_search_b2_k64_group_merge_ir` | `kernel_knn_search_k64_q128m65536_groupmerge64_kexact_0614_r27_k64thin_v1` | `normal` | 32 | 0 |
| `knn_search_b2_k64_final_merge_ir` | `kernel_knn_search_k64_q128m65536_finalmerge16_kexact_0614_r27_k64thin_v1` | `normal` | 32 | 0 |
| `knn_search_d384_q256_partial_ir` | `kernel_knn_search_d384_mma_split_partial_0612_r34_v1` | `normal` | 640 | 126720 |
| `knn_search_d384_q256_merge_ir` | `kernel_knn_search_mma_split_merge_q128_const148_v1` | `normal` | 32 | 0 |
| `knn_search_k64_q256_partial_ir` | `kernel_knn_search_80a5_blocker_k64_q256_m65536_twotile_partial_v1` | `normal` | 512 | 165120 |
| `knn_search_lowd_d256_ir` | `kernel_knn_search_mma_split_partial_v1` | `normal` | 256 | 143104 |
| `knn_search_lowd_d256_merge_ir` | `kernel_knn_search_mma_split_merge_v1` | `normal` | 32 | 0 |
| `knn_search_lowd_d256_k64_partial_ir` | `kernel_knn_search_mma_split_partial_v1` | `normal` | 256 | 143104 |
| `knn_search_lowd_d256_k64_merge_ir` | `kernel_knn_search_mma_split_merge_v1` | `normal` | 32 | 0 |
| `knn_search_lowd_dbscan_ir` | `kernel_knn_search_lowd_dbscan_d2_t128_m1536_0613_r56_cd72_v1` | `normal` | 128 | 12416 |
