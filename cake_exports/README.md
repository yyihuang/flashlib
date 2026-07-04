# Cake standalone exports

This directory contains standalone exports of the Cake upstream-main KNN
build, KNN search, and KMeans production dispatchers. Each subdirectory is an
independently installable Python/CUDA package with frozen generated CUDA,
semantic host dispatch, route inventory, correctness tests, CUPTI benchmarks,
and machine-readable validation results. None imports the Loom or Cake runtime.

Export provenance:

- upstream Cake main base: `37483b4ba16df2571f88b5f93979c20b73e14696`
- export/fix commit: `2edb9c8e413eed515bdc202cc368dfe2e49c812c`
- validation hardware: NVIDIA B200 (`sm_100a`)

| workload | frozen kernels | shapes | correctness | CUPTI | speedup vs recorded FlashLib baseline (min / geomean / max) |
| --- | ---: | ---: | ---: | ---: | ---: |
| KNN build | 232 | 112 | 112/112 | 112/112 | 0.0025x / 0.7079x / 2.0665x |
| KNN search | 520 | 163 | 163/163 | 163/163 | 0.0090x / 0.5818x / 7.8993x |
| KMeans | 356 | 228 | 228/228 | 228/228 | 2.2355x / 19.7084x / 1566.3508x |

The KNN performance regressions are intentionally reported rather than gated
away. Upstream main currently routes many search shapes through scalar
correctness-repair paths, and two KNN-build shapes require existing safe
fallbacks because their optimized B200 cubins raise illegal-instruction errors.
See each package's `VALIDATION.md`, `VALIDATION.json`, and
`BENCHMARK_RESULTS.json` for full details.
