# Correctness And Performance Results

## Export Provenance

- Package: `flashlib_cake_knn_build`
- Source repository: `ssh://internal-source-redacted
- Source commit: `21b036e2be7cdf77a3dc345e4fc8e4a5b489b143`
- Generated at: `2026-07-01T07:06:22.160199+00:00`

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
| `knn_build_stage1` | `kernel_knn_build_evolve_7bfc_split_cg2_stage1_batch8_cond4_vmin_maxtree` | `cluster` | 192 | 50432 |
