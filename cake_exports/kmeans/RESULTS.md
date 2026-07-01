# Correctness And Performance Results

## Export Provenance

- Package: `flashlib_cake_kmeans`
- Source repository: `ssh://git@gitlab-master.nvidia.com:12051/cake/cake.git`
- Source commit: `87e6e287e2a068593c719a15255fbeecdc3513f2`
- Generated at: `2026-07-01T08:43:31.541286+00:00`

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
| semantic correctness | target GPU | `python benchmarks/benchmark_shapes.py` | pending |
| kernel performance | target GPU | `python benchmarks/benchmark_shapes.py` | pending |

## Kernel Inventory

| Name | Symbol | Launch Mode | Threads | Shared Memory Bytes |
| --- | --- | --- | ---: | ---: |
| `kmeans_v10` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v10` | `normal` | 192 | 100352 |
| `kmeans_v15` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_v15` | `normal` | 192 | 133120 |
| `kmeans_d64_direct` | `kernel_flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1` | `normal` | 192 | 51200 |
| `kmeans_microdim_direct` | `kernel_flash_kmeans_assign_microdim_direct_9c0d_v1` | `normal` | 192 | 51200 |
| `kmeans_microdim_pack` | `kernel_flash_kmeans_assign_microdim_pack_6cd2_v1` | `normal` | 256 | 0 |
| `kmeans_microdim_main` | `kernel_flash_kmeans_assign_microdim_6cd2_v1` | `normal` | 192 | 51200 |
| `kmeans_lowdim_pack` | `kernel_flash_kmeans_assign_lowdim_pack_e50c_v1` | `normal` | 256 | 0 |
| `kmeans_lowdim_main` | `kernel_flash_kmeans_assign_lowdim_e50c_v1` | `normal` | 192 | 100352 |
| `kmeans_gap_pad_pack` | `kernel_flash_kmeans_assign_gap_pad_pack_v1` | `normal` | 256 | 0 |
| `kmeans_pad192_pack` | `kernel_flash_kmeans_assign_d160_pad192_pack_f9b2_v1` | `normal` | 256 | 0 |
| `kmeans_d160_padded_pack` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d160_pack_padded_b23d_v1` | `normal` | 256 | 0 |
| `kmeans_d160_splitd` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d160_splitd_v1` | `normal` | 192 | 165888 |
| `kmeans_d192_single` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d192_single_v1` | `normal` | 192 | 149504 |
| `kmeans_d192_splitd` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d192_splitd_v1` | `normal` | 192 | 198656 |
| `kmeans_d256_single` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d256_single_v1` | `normal` | 192 | 198656 |
| `kmeans_d256_splitd` | `kernel_flash_kmeans_assign_cleanroom_tcgen05_d256_splitd_v1` | `normal` | 192 | 264192 |
| `kmeans_highd_splitd` | `kernel_flash_kmeans_assign_highd_splitd_6fcf_v1` | `normal` | 192 | 51200 |
| `kmeans_highd_splitk_partial` | `kernel_flash_kmeans_assign_highd_splitk_partial_8de8_v1` | `normal` | 192 | 51200 |
| `kmeans_highd_splitk_reduce` | `kernel_flash_kmeans_assign_highd_splitk_reduce_8de8_v1` | `normal` | 128 | 0 |
| `kmeans_highd_splitk_blockn64_g2r4_partial` | `kernel_flash_kmeans_assign_highd_splitk_partial_blockn64_g2r4_b5a6_v1` | `normal` | 192 | 43008 |
| `kmeans_highd_splitk_blockn64_g2r4_reduce` | `kernel_flash_kmeans_assign_highd_splitk_reduce_blockn64_g2r4_b5a6_v1` | `normal` | 256 | 0 |
| `kmeans_highd_splitk_blockn64_g1r4_partial` | `kernel_flash_kmeans_assign_highd_splitk_partial_blockn64_g1r4_streamdep_r63_v1` | `normal` | 192 | 43008 |
| `kmeans_highd_splitk_blockn64_g1r4_reduce` | `kernel_flash_kmeans_assign_highd_splitk_reduce_blockn64_g1r4_streamdep_r63_v1` | `normal` | 256 | 0 |
| `kmeans_highd_paired_xreuse_r47_partial` | `kernel_flash_kmeans_assign_highd_paired_xreuse_dualtmem_producer_r47_v1` | `normal` | 192 | 75776 |
| `kmeans_highd_paired_ownerreduce_r39_reduce1` | `kernel_flash_kmeans_assign_highd_paired_ownerreduce_r39_reduce1_unroll_v1` | `normal` | 64 | 0 |
| `kmeans_highd_paired_packedpartial_r2_partial` | `kernel_flash_kmeans_assign_highd_paired_packedpartial_producer_7b3c_v1` | `normal` | 192 | 43008 |
| `kmeans_highd_paired_packedpartial_r2_reduce` | `kernel_flash_kmeans_assign_highd_paired_packedpartial_reduce_r2_7b3c_v1` | `normal` | 128 | 0 |
