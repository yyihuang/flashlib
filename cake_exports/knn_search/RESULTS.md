# Correctness And Performance Results

## Export Provenance

- Package: `flashlib_cake_knn_search`
- Source repository: `ssh://git@gitlab-master.nvidia.com:12051/cake/cake.git`
- Source commit: `21b036e2be7cdf77a3dc345e4fc8e4a5b489b143`
- Generated at: `2026-07-01T07:06:34.097711+00:00`

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
| `knn_search_q1_tile_reduce_partial_v1` | `kernel_knn_search_q1_tile_reduce_partial_v1` | `normal` | 256 | 5120 |
| `knn_search_q1_tile_reduce_merge_v1` | `kernel_knn_search_q1_tile_reduce_merge_v1` | `normal` | 256 | 640 |
