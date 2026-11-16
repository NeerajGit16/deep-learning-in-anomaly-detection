# Decision point: are the links between sensors curved once a plant runs at several settings?

Written 16 November 2026, after notebooks `05_cranfield.ipynb` and `06_multimode_tep.ipynb`.
This file reports numbers against the rule. It does not make the decision.

## The rule (fixed before either dataset was measured)

- **GO** if, on either the Cranfield rig or the six-mode chemical plant, the curved share's whole 90% range lies above +3 points.
- **NO-GO** if both stay at or near zero.

The measure is the **pooled, same-moment curved share**: all settings (or modes) put together, each sensor predicted from the other sensors' readings at the same instant, on runs or visits the models never saw. The curved share is how many more points of a sensor's movement a flexible model explains than a straight-line model does. Lagged versions and "setting told to the straight model" are supporting numbers only.

## The numbers

| Dataset | Curved share (points) | 90% range | Against +3 points | Sensors gaining over 10 points |
|---|---|---|---|---|
| **Cranfield flow rig, runs T1 + T2 (headline)** | **+2.8** | **+1.3 to +3.7** | range straddles +3 | 3 of 23 |
| Cranfield, all three runs T1 to T3 | +2.2 | +1.8 to +2.3 | wholly below +3 | 3 of 23 |
| **Six-mode chemical plant (headline)** | **+0.07** | **+0.04 to +0.10** | wholly below +3 | 0 of 52 |

Cranfield has two rows because of the outlier-run rule: run T3 holds more than half of the movement of eight sensors (a warmer day, a lower separator level), so the version without it is the headline and the version with it is shown beside it. The ranges above come from resampling runs. Cranfield has only two or three runs, so its range was also computed by resampling visits to a flow setting: +1.9 to +3.8 for T1 + T2 (76 visits) and +1.5 to +2.9 for T1 to T3 (117 visits). Both readings land on the same side.

## Which side of the rule

- **GO condition (whole range above +3 on either dataset): not met.** Cranfield's headline range reaches above +3 at its top end but starts at +1.3. The six-mode plant's range ends at +0.10.
- **NO-GO condition (both at or near zero):** the six-mode plant is at zero. Cranfield is not at zero: its curved share is about +2 to +3 points and its range excludes zero. So Cranfield sits **between** the two conditions, in the gap the rule does not cover.

## Supporting numbers

| | Cranfield, T1 + T2 | Six-mode plant |
|---|---|---|
| Straight share (0 to 1) | 0.81 | 0.90 |
| Setting or mode told to the straight model | +1.1 (+0.4 to +1.5) | +0.03 (+0.01 to +0.04) |
| With lags | +2.2 falling to 0.0 as more history is added | +0.05 falling to +0.02 |
| Inside single settings or modes | about 0 (middle setting 0.0, largest +5) | −0.9 to +1.9, every range below +3 |
| Other hold-out | blocks of time: +2.5 (+1.3 to +3.0) | with 93 mode-transition runs added: +0.41 (+0.33 to +0.50) |

## What to keep in mind when reading these

1. **On Cranfield, more than half of the curved share is level shifts between settings.** Telling the straight model which flow setting the rig is in cuts the curved share from +2.8 to +1.1 points. The three sensors that gain over 10 points show clusters, a meter sitting at its floor, a link that forks, and a reading that flips between two states. None shows a smooth curve (notebook 05, section 9).
2. **On the six-mode plant, the pooled measure could hardly have said yes.** For 41 of 51 sensors, over 99% of the pooled movement is the gap between modes, and a straight model with 51 inputs can hit six separate patches exactly whatever shape they make. The measures that could have shown curvature there (transition runs, inside each mode) are small too (notebook 06, section 8).
3. **The six-mode plant does have real curvature in a few places.** In modes 3 and 6, valves reach their limits and a control rule switches from one valve to another. Those sensors gain 18 to 60 points from curves, as corners and ceilings (notebook 06, section 10). Averaged over all sensors this is +0.5 and +1.9 points in those two modes.
4. **The evidence is thin in places.** Cranfield has three normal runs. The six-mode files hold only 5 different fault-free runs per mode (60 are stored, 12 copies of each).

## Detectors, for context (not part of the rule)

- **Cranfield:** at 1 false alarm in 100, the straight-link detector catches 0.38 of faulty readings (0.27 to 0.51) and the curved-link detector 0.37. The gap is 1.4 points, far under the 12-point margin: a tie. Both are ahead of the other six detectors over all runs.
- **Six-mode plant:** detectors have not been run. They need faulty runs, which have not been downloaded (about 210 MB for one run per fault per mode).

## Not decided here

Whether this is GO, NO-GO, or a reason to look at a different kind of plant. That call is left to the people making it.
