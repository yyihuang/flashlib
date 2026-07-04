## Pre-publication GPU validation: PASS

- Hardware: `NVIDIA B200` (`sm_100a`)
- Shapes: correctness `112/112`, route fidelity `112/112`, CUPTI `112/112`
- Public speedup vs recorded FlashLib: min `0.2314x`, geomean `0.7296x`, max `2.0338x`
- Prepared speedup: min `0.8233x`, geomean `2.0137x`, max `4.2444x`; `111/112` shapes exceed `1x`
- Public/prepared GPU-span ratio: geomean `2.7601x`
- Source: Cake main `90083f7caff1734b740c24083f944615d7fab15e`
- Full summary: [`VALIDATION.json`](VALIDATION.json); rows: [`BENCHMARK_RESULTS.json`](BENCHMARK_RESULTS.json)
