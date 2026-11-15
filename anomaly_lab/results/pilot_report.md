# Baseline pilot

Primary metric: share of anomalous samples flagged when each method's threshold gives 1% false alarms on normal samples (random scoring gets 0.01). Intervals resample runs 1,000 times.

## SKAB (34 runs, 54% of scored samples anomalous; one threshold per run, averaged over runs)

| Method | DR @1% FA | 90% interval | DR @5% FA | Deployable FA | Deployable DR | Events flagged | Median delay (s) | AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Straight-link residual | 0.644 | 0.55-0.73 | 0.677 | 0.019 | 0.293 | 0.41 | 36 | 0.884 |
| Curved-link residual | 0.609 | 0.51-0.70 | 0.641 | 0.019 | 0.291 | 0.44 | 52 | 0.875 |
| Matrix Profile | 0.562 | 0.44-0.67 | 0.578 | 0.024 | 0.218 | 0.26 | 18 | 0.814 |
| Rolling z-score | 0.496 | 0.38-0.60 | 0.518 | 0.019 | 0.218 | 0.26 | 14 | 0.797 |
| EWMA chart | 0.496 | 0.38-0.60 | 0.520 | 0.020 | 0.218 | 0.26 | 15 | 0.797 |
| PCA (T2 + SPE) | 0.412 | 0.30-0.51 | 0.464 | 0.021 | 0.218 | 0.26 | 13 | 0.797 |
| Isolation Forest | 0.237 | 0.18-0.33 | 0.370 | 0.013 | 0.096 | 0.48 | 78 | 0.778 |
| CUSUM chart | 0.083 | 0.02-0.16 | 0.087 | 0.016 | 0.015 | 0.03 | 200 | 0.550 |

Random-score AP on this dataset is about 0.54.
A gap between two methods under **0.12** is not called a win (fixed margin). For information, the measured chance wobble of a gap between two methods here: typical 0.049, largest 0.068 (bootstrap s.d.).

## TEP (21 runs, 83% of scored samples anomalous; one threshold, set on 1 separate fault-free run(s) over the same hours of a run as the fault)

| Method | DR @1% FA | 90% interval | Old rule (first 8 hours) | DR @5% FA | Deployable FA | Deployable DR | Events flagged | Median delay (s) | AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Straight-link residual | 0.811 | 0.68-0.92 | 0.812 | 0.822 | 0.009 | 0.811 | 1.00 | 1080 | 0.974 |
| Curved-link residual | 0.806 | 0.68-0.92 | 0.812 | 0.830 | 0.006 | 0.807 | 1.00 | 1260 | 0.973 |
| Matrix Profile | 0.723 | 0.60-0.84 | 0.765 | 0.783 | 0.004 | 0.724 | 1.00 | 3780 | 0.974 |
| CUSUM chart | 0.695 | 0.56-0.82 | 0.711 | 0.719 | 0.006 | 0.695 | 0.86 | 9180 | 0.988 |
| PCA (T2 + SPE) | 0.643 | 0.50-0.78 | 0.648 | 0.698 | 0.007 | 0.642 | 1.00 | 2340 | 0.967 |
| Rolling z-score | 0.601 | 0.46-0.74 | 0.617 | 0.623 | 0.006 | 0.601 | 1.00 | 2340 | 0.967 |
| EWMA chart | 0.590 | 0.45-0.74 | 0.611 | 0.609 | 0.004 | 0.591 | 0.95 | 4140 | 0.968 |
| Isolation Forest | 0.402 | 0.28-0.56 | 0.408 | 0.475 | 0.013 | 0.411 | 1.00 | 8064 | 0.953 |

Random-score AP on this dataset is about 0.83.
A gap between two methods under **0.12** is not called a win (fixed margin). For information, the measured chance wobble of a gap between two methods here: typical 0.062, largest 0.086 (bootstrap s.d.).
