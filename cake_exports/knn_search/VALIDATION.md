# B200 Export Validation

Status: correctness, route, runtime, and evidence gates pass; the per-shape
performance floor remains open.

## Provenance

- Cake repository: `ssh://git@gitlab-master.nvidia.com:12051/averyh/cake.git`
- Cake commit: `0e96313db7754bc32e9e3e75ffe04a454ae44450`
- Hardware: NVIDIA B200 (`sm_100a`)
- Canonical array: `729920_[0-3]`, four final tasks completed `0:0`
- Package artifact SHA-256:
  `f52e46daa0b8db39b97c4b1dec630014e7b8aa9e9b152e14887a172325c73413`
- Evidence artifact SHA-256:
  `eac43b5428b7c42290d7c92bdbccaa71c19e59dd1350554019a73193fd0c0d2f`

The B200 package was generated directly from that commit. Only provenance
metadata was normalized from a temporary bundle path to the canonical Cake
remote. A byte-for-byte hash comparison confirms every generated CUDA, Python,
benchmark, test, and route/shape source file matches the B200 artifact.

## Fail-Closed Gates

| Gate | Result |
| --- | ---: |
| Candidate correctness | 163/163 |
| Live `flashlib.flash_knn` correctness | 163/163 |
| Expected route match | 163/163 |
| Exact prepared launch plan | 163/163 |
| Non-empty shared baseline/public/prepared session | 163/163 |
| CUPTI timing for candidate and live baseline | 163/163 |
| Distinct correlated launch/kernel counts | 163/163 |
| Host enqueue, synchronized E2E, and cold fields | 163/163 |
| Prepared launch activity equals captured plan | 163/163 |
| Legacy `activity_count` equals kernel activity count | 163/163 |

All six deterministic measurement-order permutations are represented. The
official metric is cold-L2 CUPTI correlated GPU span; no historical timing is
used as a performance denominator.

## Runtime And Stream Safety

- All 367 vendored `_dispatch` modules: zero internal synchronize sites and
  zero unexpected host-readback sites.
- Source gate: 91 passed; generated prepared-API gate: 3 passed; KNN-search GPU
  e2e: 1 passed.
- K11, K64, ordinary multi-kernel, and D63 scalar plans: recall 1.0 on two
  distinct streams and cross-stream replay rejection.
- D63 scalar and direct-source D384 cache owner: distinct stream keys and
  distinct distance/index scratch pointers.
- K11 correctness-only guard: exact mutation-replay indices, caller-owned
  outputs, four captured launches, and no performance baseline claim.

`PACKAGE_STATIC_AUDIT.json`, `GPU_CERTIFICATE.json`, and `GPU_E2E.log` preserve
the raw evidence.

## Performance

| Path / ratio | Min | Median | Geomean | Max | `<1.0x` | `<1.2x` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Public / live FlashLib | 0.02494x | 2.0101x | 1.3028x | 6.2471x | 40 | 43 |
| Prepared / live FlashLib | 0.02494x | 2.0226x | 1.3087x | 6.3289x | 40 | 43 |
| Public span / prepared span | 0.9190x | 1.0041x | 1.0045x | 1.0260x | — | — |

Host enqueue median is `210.561 us` public and `12.952 us` prepared; the
per-shape ratio geomean is `15.5543x`. Synchronized E2E median is `372.177 us`
public and `147.148 us` prepared; ratio geomean is `2.3292x`. This confirms the
prepared API removes material host work while device span is unchanged.

The 40 sub-1.0x rows comprise 37 scalar correctness-repair routes, one round34
tail wrapper, and two optimized routes. Prepared inter-kernel gaps are only
`0.449401 ms / 93.343686 ms` (`0.4814%`) in aggregate, and 39 rows remain below
1.0x using kernel sum alone. The remaining floor gap is primarily device
kernel work, not dispatch/FFI or launch gaps.
