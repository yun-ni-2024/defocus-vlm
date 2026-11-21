# Evaluation

## Test images

All 800 text image pairs of the test set: 20 defocus steps, 40 per step (20 Times, 20 Verdana). The test set is used only here, never for training or tuning. See `data.md`.

## Score per image

The VLM transcribes the photograph, and the transcription is compared with the ground truth by the **character accuracy**:

$$
\text{accuracy} = \max\left(0,\ 1 - \frac{\text{Levenshtein}(\text{prediction}, \text{ground truth})}{\text{length of ground truth}}\right)
$$

- The Levenshtein (edit) distance is the smallest number of single-character insertions, deletions and substitutions that turn one string into the other.
- 1 means an exact match; 0 means nothing useful was read. Long invented outputs can make the distance larger than the ground truth, which is why the score is clipped at 0.
- Whitespace: currently all spaces and line breaks are removed before comparing (`keep_whitespace: false`). This is not final.

Example (test step 7, `focusStep_7_timesR_size_30_sample_0001`): ground truth `beZtuqe4du / ZagPvxR6Ys / ThSZ AKYWe`, prediction `Instagram / TagPeopleTo / ThisActing`, accuracy 0.17.

## Score per step

For each defocus step, the 40 scores are averaged. The uncertainty of that mean is a 95% bootstrap confidence interval: resample the 40 scores with replacement 2,000 times, average each resample, and take the 2.5th and 97.5th percentiles of these averages. The bootstrap is used because the scores are bounded by 0 and 1 and often pile up at either end, so a normal approximation does not fit. The random seed is fixed, so the interval is reproducible.

Each method thus gives one curve: accuracy against defocus step, with error bars.

## Reference curves

| Curve | What it shows |
|---|---|
| Sharp reference | the sharp photographs read by the same VLM: the best any method can reach |
| (a) baseline | the blurred photographs read directly |
| Calibrated-PSF reference | deconvolution with the PSF measured from the calibration targets: an upper bound for (b), since no test-time method is told the PSF |

## What is reported

From the proposal:

1. the accuracy of every method at every defocus step, with confidence intervals;
2. for (c), training on real versus simulated pairs; for (d), VLM-feature loss versus pixel loss;
3. the kinds of errors each method makes: refusals, invented words, wrong characters.
