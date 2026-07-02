## Pre-publication GPU validation: FAILED

- Hardware: `NVIDIA GB300` (`sm_103a`), Slurm job `1449292`
- Declared shapes: `58` Cake-recorded build rows
- Correctness: failed on multiple rows; run stopped before completion
- CUPTI benchmark: **not run** because correctness did not pass
- Cause: the exported Python interface is still the old D=128/limited-K dispatcher and does not consume the full exported CUDA portfolio
- Full summary: [`VALIDATION.json`](VALIDATION.json)
