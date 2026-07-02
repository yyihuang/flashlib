## Pre-publication GPU validation: FAILED

- Hardware: `NVIDIA GB300` (`sm_103a`), Slurm job `1449293`
- Declared shapes: `163` Cake-recorded search rows
- Correctness: `73/163` passed, `90/163` failed
- CUPTI benchmark: **not run** because correctness did not pass
- Cause: the exported Python interface supports only D=128 and K<=10 while the exported CUDA inventory and Cake ledger contain broader routes
- Full summary: [`VALIDATION.json`](VALIDATION.json)
