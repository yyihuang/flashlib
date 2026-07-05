# B200 Export Validation

- Exact Cake source: `c4dae7228441b389d379e47135da23eb22dc96a7` (!249)
- Export/full-portfolio job: NSC `726906`
- Contract rows: 228/228; unique tensor shapes: 225
- Candidate correctness: 228/228
- Expected route parity: 228/228
- Pinned `07cf2a27928aacf6790c950a265d8b8dc83c87cf` tie-inclusive correctness: 228/228
- Exact 07cf index agreement: 223/228; five equal-distance ties
- Same-session baseline/public/prepared provenance: 228/228
- Prepared speedup: min 0.6711x, geomean 1.5574x, median 1.4744x, max 12.5338x
- Prepared rows below 1x / 1.2x: 39/228 / 82/228
- Public speedup: min 0.5925x, geomean 1.5227x, median 1.4391x, max 12.6269x
- Public rows below 1x / 1.2x: 51/228 / 85/228
- Generated package tests: 239/239 passed
- Timing: same-process, per-shape SHA-permuted CUPTI GPU span with cold-L2 flushing on NVIDIA B200

Two independent B200 A/B sessions retained five exact routes: four small
D16/D32 pad-to-D64 routes and one direct D224 route. Three neutral/slower B8
micro-D candidates were rejected and remain on the incumbent route.

The remaining 39 prepared rows below 1x comprise 28 device/kernel-sum deficits
and 11 gap-only crossings. Prepared kernel sum is 7.5966 ms and correlated
gaps are 0.2355 ms (3.01% of total span). The full public/prepared/baseline
host and GPU decomposition is recorded per row in
[`BENCHMARK_RESULTS.json`](BENCHMARK_RESULTS.json).
