# Baseline pilot

Primary metric: share of anomalous samples flagged when each method's threshold gives 1% false alarms on normal samples (random scoring gets 0.01). Intervals resample runs 1,000 times.

## SKAB (34 runs, 54% of scored samples anomalous; one threshold per run, averaged over runs)

| Method | DR @1% FA | 90% interval | DR @5% FA | Deployable FA | Deployable DR | Events flagged | Median delay (s) | VUS-PR | AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Matrix Profile | 0.562 | 0.44-0.67 | 0.578 | 0.024 | 0.218 | 0.26 | 18 | 0.817 | 0.814 |
| Rolling z-score | 0.496 | 0.38-0.60 | 0.518 | 0.019 | 0.218 | 0.26 | 14 | 0.800 | 0.797 |
| EWMA chart | 0.496 | 0.38-0.60 | 0.520 | 0.020 | 0.218 | 0.26 | 15 | 0.800 | 0.797 |
| PCA (T2 + SPE) | 0.412 | 0.30-0.51 | 0.464 | 0.021 | 0.218 | 0.26 | 13 | 0.800 | 0.797 |
| Isolation Forest | 0.237 | 0.18-0.33 | 0.370 | 0.013 | 0.096 | 0.48 | 78 | 0.782 | 0.778 |
| CUSUM chart | 0.083 | 0.02-0.16 | 0.087 | 0.016 | 0.015 | 0.03 | 200 | 0.556 | 0.550 |

Random-score AP on this dataset is about 0.54.
Paired-gap noise: median bootstrap s.d. 0.058, largest 0.068. Margin = max(0.05, 2 x median) = **0.116**.

## TEP (21 runs, 83% of scored samples anomalous; one threshold across all runs)

| Method | DR @1% FA | 90% interval | DR @5% FA | Deployable FA | Deployable DR | Events flagged | Median delay (s) | VUS-PR | AP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Matrix Profile | 0.765 | 0.61-0.89 | 0.819 | 0.004 | 0.724 | 1.00 | 3780 | 0.974 | 0.974 |
| CUSUM chart | 0.711 | 0.58-0.85 | 0.786 | 0.006 | 0.695 | 0.86 | 9180 | 0.989 | 0.988 |
| PCA (T2 + SPE) | 0.648 | 0.51-0.79 | 0.707 | 0.007 | 0.642 | 1.00 | 2340 | 0.968 | 0.967 |
| Rolling z-score | 0.617 | 0.47-0.77 | 0.683 | 0.006 | 0.601 | 1.00 | 2340 | 0.967 | 0.967 |
| EWMA chart | 0.611 | 0.47-0.76 | 0.667 | 0.004 | 0.591 | 0.95 | 4140 | 0.969 | 0.968 |
| Isolation Forest | 0.408 | 0.27-0.56 | 0.507 | 0.013 | 0.411 | 1.00 | 8064 | 0.954 | 0.953 |

Random-score AP on this dataset is about 0.83.
Paired-gap noise: median bootstrap s.d. 0.061, largest 0.083. Margin = max(0.05, 2 x median) = **0.122**.
