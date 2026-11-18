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

## Not decided here (as first written)

Whether this is GO, NO-GO, or a reason to look at a different kind of plant. That call was left to the people making it; it was made on 16 November (below).

## Decision (16 November 2026)

**Made by the user: not a go by the rule.** Neither dataset's curved share had its whole range above +3 points. The six-mode plant is at zero; Cranfield sits in the gap the rule does not cover (above zero, range reaching +3 only at its top end).

**Instead, the project changes direction:**

1. **No big simulator study for now.** A small one comes later, to show what a clean "yes" looks like.
2. **A new main question:** does the curved share, measured on normal data alone, predict when flexible and deep models beat straight-line ones? It is tested on real datasets.
3. **Deep models start now** (notebook 09).
4. **IndPenSim replaces the Pensim benchmark.** Pensim needs a request form, and its licence forbids sharing the data, so others could not rerun the work. IndPenSim is public (CC BY 4.0).
5. **Pretrained "foundation" models are put off** for now.

**Why.** The six-mode plant was near zero (+0.07). Cranfield's +2.8 was mostly level shifts between settings (+1.1 once the straight model is told the setting). The metal etcher gave the first "yes" (+8.1, notebook 07), but only a small gain in detection: the curved-link detector caught 4.2 points more than the straight-link one, a tie under the 12-point margin.

### Is Cranfield near the same "islands" limit as the six-mode plant?

On the six-mode plant, almost all of each sensor's movement in the pooled data is the jump between modes. A straight model with many inputs can then hit every mode's average exactly, whatever shape the modes make, so the pooled curved share cannot show a curve between them (notebook 06, section 8). Cranfield has 20 flow settings and, for each sensor, 22 other sensors to predict it from. So in principle a straight model has enough freedom to place each setting's average too.

The share of each sensor's movement that lies between the flow settings (runs T1 + T2, the readings taken at a setting; `cranfield.between_table()`):

| | Cranfield (T1 + T2) | Cranfield (T1 to T3) | Six-mode plant |
|---|---|---|---|
| Sensors with over 99% of movement between settings or modes | 5 of 23 | 5 of 23 | 41 of 51 |
| Sensors with over 90% | 11 of 23 | 9 of 23 | 45 of 51 |
| Sensors with over half | 14 of 23 | 13 of 23 | 46 of 51 |
| The middle sensor | 0.86 | 0.79 | 1.00 |

Over 99% between settings: the water and air flows in, two valve positions and the pressure drop along the riser, which are the rig's settings and what follows directly from them. Nearly all within settings: the separator pressure PT501 (0.05), the water pump current, the top separator level and the temperatures.

**How this affects reading the +2.8.** Cranfield is **about halfway** to the six-mode limit. For roughly half of its sensors, the pooled measure says little about how a link bends *between* settings, because the settings are islands a straight model can place one by one. For those sensors the +2.8 is a weak test of curvature between settings. It could miss a real bend there, and what it does find is not evidence of one. For the other half, the measure works as intended.

What the +2.8 does contain is mostly level differences between visits to the same setting: telling the straight model the setting brings it down to +1.1. Measured inside each setting, it is about zero (notebook 05). Of the three sensors that gained over 10 points, two are island sensors: the flow out of the top separator (0.92 of its movement between settings) and the flow at the top of the riser (0.69). The third, the separator pressure PT501 (0.05), is not.

### Detectors on the six-mode plant (added 16 November, notebook 06 sections 13 to 18)

The 168 faulty runs (one per fault per mode) were scored by detectors fitted on 24 normal runs, with one alarm line set on 6 other normal runs over the hours the faults cover. The straight-link and curved-link detectors tie, catching 0.46 of faulty readings each (ranges 0.41 to 0.52). The gap is within half a point of zero in every mode, including modes 3 and 6, where valves hit their limits. Both are ahead of every other detector (Matrix Profile is next at 0.32). False alarms before the faults are about 1 in 100, so the line carries over. On this plant, the curves that exist do not help catch faults.

## Where the hypotheses stand (18 November 2026, after round 5)

Each line: the hypothesis, where it stands, and the evidence. Numbers are catch rates at 1 false alarm in 100; a win needs 12 points with a range clear of zero. H1 to H4 are the four hypotheses of `main.ipynb`; the lines after them came up along the way.

| Hypothesis | Where it stands | Evidence |
|---|---|---|
| **H1.** For faults in a single sensor, simple checks are as good as deep learning | **Supported** | On faults where one sensor moves on its own, no deep model beats a per-sensor check (rolling z-score, EWMA or CUSUM) on the water pump or the chemical plant. (The positive control's faults also move one sensor, but inside its normal range, so they show only in the links: H3) |
| **H2.** For broken straight-line links, PCA is as good as deep learning | **Supported** | On the positive control's flat data, PCA (10 sensors) and the straight-link residual (40 sensors) catch every fault at every size; with 10 sensors the autoencoder matches PCA from 10,000 readings (the PCA-plus-correction autoencoder, which starts as PCA, from 2,000); with 40 sensors only the LSTM matches the straight-link residual, at 50,000. On the four real datasets with little curve (+3 points or less), the best deep model ties or trails a straight-line detector |
| **H3.** The more curved or mode-dependent the links, the further ahead deep learning gets | **Not on real data; narrowly on data built for it** | On real data no deep model beats the best simple detector, and the two most curved datasets are where memory checks lead. On the positive control's second version, deep models win only on bent data with 10 sensors and 50,000 readings (3 of 54 cells; first win +37 points, range +6 to +62); with 40 sensors the curved-link residual leads, and the win does not grow from bend 1 to bend 2 |
| **H4.** Something measured in advance (first a data simulator; since 16 November, the curved share on normal data) says which method will win on real data | the curved share predicts gains over the straight-link detector (11-12 of 15 pairs) but not over the best simple detector (3 of 15); the repeat share does not predict where memory checks win (7 of 15). | Notebook 10, sections 5 to 7; unchanged with the deep models trained to the end. Six datasets: a picture, not a test |
| Links between sensors bend once a plant runs at several settings | **Partly.** Clearly on two batch processes, not on continuous plants | Curved share +8 (metal etcher) and +15 (penicillin batches); +3 or less elsewhere. The bends are mostly corners, steps and switches of the control system; a straight model allowed a few corners recovers half to two thirds |
| Flexible models of the links catch more faults than straight ones | **Not supported** | The curved-link and straight-link detectors tie on all six datasets (within 4 points), also after the fold audit (six-mode plant +3) |
| Deep models beat simple detectors on plant data | **Not supported, also when trained to the end** | No deep model beats the best simple detector on any of the six datasets or the 11 generated faults. Behind by the margin on the flow rig (−15) and three generated faults; the PCA-plus-correction autoencoder turns the penicillin batches (−9) and the base-flow fault (−10) into ties. The etcher's round-4 gap (−61) rested on edited columns; without them the best deep model ties the CUSUM chart (+4 and −1) |
| The deep models were trained long enough | **No, but it did not matter** | 287 of 575 trainings under notebook 09's budget used all 100 passes. Trained to the end (up to 1,000 passes; none needed all of them), 11 of 42 deep results move by more than 2 points and no verdict against the best simple detector changes |
| The autoencoder's win on the penicillin batches was depth | **Rejected** | It was knowing the recipe's schedule: the unseen value check ties it (0.645 against 0.639) and the nearest normal reading on all sensors beats it (0.78) |
| A deep model wins where it should (positive control) | **Yes, narrowly** | Version 2 (every sensor bent its own way, faults of three sizes, training to the end; the 40 sensors split into four groups and the bend enlarged after the check on normal data): first deep win at bend 1, 10 sensors, 50,000 readings, half a spread, the PCA-plus-correction autoencoder 0.42 against the curved-link residual's 0.05, ahead of every simple detector. 3 of 54 cells; none with 40 sensors or fewer than 50,000 readings. Version 1: none at 24 settings |
| Memory checks win where normal behaviour repeats exactly | **Partly; the repeat share does not predict it** | They lead on the penicillin batches (nearest normal reading 0.78), where faults move a stepped setting to a value never seen. The etcher's 0.95 was an artefact of the edited setting columns (0.16 to 0.17 without them). The repeat share orders the datasets like the memory checks' gain in 7 of 15 pairs |
| Recent history (a forecaster) beats links on the six-mode plant | **Corrected** | Mostly a fold artefact: with balanced folds the link detectors catch 0.67 to 0.71 and the straight forecaster 0.69, a tie |
| Alarm lines set on separate normal data carry over to the faulty runs | **Supported** | Before the fault, 1 of about 120 detector-and-dataset pairs exceeds 3% false alarms (isolation forest on the penicillin batches, 4.7%) |

## Where the hypotheses stood (17 November 2026, after round 4; kept for the record)

Each line: the hypothesis, where it stands, and the evidence. Numbers are catch rates at 1 false alarm in 100; a win needs 12 points with a range clear of zero.

| Hypothesis | Where it stands | Evidence |
|---|---|---|
| Links between sensors bend once a plant runs at several settings | **Partly.** Clearly on two batch processes, not on continuous plants | Curved share +8 (metal etcher) and +15 (penicillin batches); +3 or less elsewhere. The bends are mostly corners, steps and switches of the control system; a straight model allowed a few corners recovers half to two thirds |
| Flexible models of the links catch more faults than straight ones | **Not supported** | The curved-link and straight-link detectors tie on all six datasets (within 4 points), also after the fold audit (six-mode plant +3) |
| Deep models beat simple detectors on plant data | **Not supported** | No deep model beats the best simple detector on any of the six datasets or the 11 generated faults; on three datasets the best deep model is behind by the margin (etcher −61, flow rig −14, penicillin batches −14) |
| The autoencoder's win on the penicillin batches was depth | **Rejected** | It was knowing the recipe's schedule: the unseen value check ties it (0.645 against 0.639) and the nearest normal reading on all sensors beats it (0.78) |
| A deep model wins where it should (positive control) | **Not shown with this setup** | On data built for a deep win, no deep model beat the best simple detector at any of 24 settings. The autoencoder needed 10,000 readings to match PCA on a flat surface and 50,000 to learn a bent one, where a link detector matched it. A limit of notebook 09's deep design, and of the generator (with 40 sensors its bends were flat for a straight line through the other sensors) |
| Memory checks win where normal behaviour repeats exactly | **Partly; the repeat share does not predict it** | Memory checks win clearly on the etcher (unseen value check 0.95) and lead on the penicillin batches (nearest normal reading 0.78), where faults move a stepped setting to a value never seen. The repeat share is 0.96 to 1.00 on four datasets, and memory checks lose on three of them; it orders the datasets like the memory checks' gain in 7 of 15 pairs |
| The curved share on normal data says in advance where flexible models gain | **Only against the straight-link detector** | Same order in 11 or 12 of 15 pairs for the gains over the straight-link detector; 3 of 15 for the gain over every simple detector. Six datasets: a picture, not a test |
| Recent history (a forecaster) beats links on the six-mode plant | **Corrected** | Mostly a fold artefact: with balanced folds the link detectors catch 0.67 to 0.71 and the straight forecaster 0.69, a tie |
| Alarm lines set on separate normal data carry over to the faulty runs | **Supported** | Before the fault, 1 of about 120 detector-and-dataset pairs exceeds 3% false alarms (isolation forest on the penicillin batches, 4.7%) |
