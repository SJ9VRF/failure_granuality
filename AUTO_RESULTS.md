# Auto-generated Results

> Generated directly from CSV outputs. Do not edit numerical values by hand.

## Main heterogeneous-distribution study

| Method | Held-out | External shift |
|---|---:|---:|
| difficulty | 87.89% | 84.01% |
| random_gran | 87.83% | 83.96% |
| capability | 87.81% | 83.96% |
| uniform | 87.81% | 83.92% |
| family | 87.76% | 83.87% |
| failure | 87.74% | 83.86% |

## Direct causal granularity intervention (100 untouched seeds)

| Granularity | Support entropy | Coverage | State repair | Target-only success | Broad held-out | External |
|---|---:|---:|---:|---:|---:|---:|
| global | 0.582 | 0.133 | 0.396 | 72.30% | 72.47% | 67.46% |
| capability | 0.475 | 0.075 | 0.583 | 83.47% | 63.51% | 58.93% |
| family | 0.258 | 0.038 | 0.592 | 81.97% | 50.38% | 45.61% |
| failure | 0.202 | 0.035 | 0.594 | 81.97% | 49.15% | 43.93% |
- Failure − Global, support_entropy: -0.380, 95% bootstrap CI [-0.386, -0.374]; paired t p=1.08e-109.
- Failure − Global, state_repair: 0.197, 95% bootstrap CI [0.182, 0.212]; paired t p=5.08e-46.
- Failure − Global, heldout: -23.320 pp, 95% bootstrap CI [-23.968, -22.668] pp; paired t p=1.62e-85.
- Failure − Global, external: -23.526 pp, 95% bootstrap CI [-24.186, -22.862] pp; paired t p=9.54e-85.
- Failure − Global, target_success: 9.664 pp, 95% bootstrap CI [8.424, 10.944] pp; paired t p=2.16e-27.

## Empirical phase map: fraction of 20 transfer × prevalence cells won

| Diagnostic noise | Budget | Global | Capability | Family | Failure |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 40 | 0.00 | 0.20 | 0.00 | 0.80 |
| 0.0 | 80 | 0.15 | 0.65 | 0.05 | 0.15 |
| 0.0 | 160 | 1.00 | 0.00 | 0.00 | 0.00 |
| 0.2 | 40 | 0.00 | 0.20 | 0.00 | 0.80 |
| 0.2 | 80 | 0.00 | 0.80 | 0.20 | 0.00 |
| 0.2 | 160 | 0.80 | 0.20 | 0.00 | 0.00 |
| 0.4 | 40 | 0.00 | 0.20 | 0.00 | 0.80 |
| 0.4 | 80 | 0.00 | 0.70 | 0.25 | 0.05 |
| 0.4 | 160 | 0.80 | 0.20 | 0.00 | 0.00 |

## Alternative support definitions

| Granularity | Normalized entropy | Effective support | Coverage | JS from global | Broad eval |
|---|---:|---:|---:|---:|---:|
| global | 0.584 | 0.100 | 0.134 | 0.000 | 72.23% |
| capability | 0.474 | 0.054 | 0.073 | 0.524 | 68.19% |
| family | 0.257 | 0.016 | 0.038 | 0.671 | 58.03% |
| failure | 0.206 | 0.012 | 0.035 | 0.706 | 57.44% |