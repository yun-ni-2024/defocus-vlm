# Evaluation

## Metric

The VLM transcribes each test text image. The score is the **character accuracy**:

$$
\text{accuracy} = 1 - \frac{\text{edit distance(prediction, ground truth)}}{\text{length of ground truth}}
$$

clipped at 0. The edit distance is the Levenshtein distance: the smallest number of single-character insertions, deletions and substitutions that turn one string into the other.

Open point: the pilot and the GPU check of 2026-11-08 removed all whitespace before comparing. Whether the final evaluation keeps spaces and line breaks will be fixed in the evaluation code and recorded here.

## What is reported

From the proposal:

1. the character accuracy of each method at each defocus step of the test set, with confidence intervals;
2. for (c), real versus simulated training pairs; for (d), VLM-feature loss versus pixel loss;
3. the kinds of errors each method makes: refusals, invented words, wrong characters.

## Confidence intervals

Per step, the interval is a bootstrap over the step's test images: resample the per-image accuracies with replacement 2000 times, take the mean of each resample, and report the 2.5% and 97.5% quantiles of those means (seed fixed, `src/eval/summary.py`).

## Test set size

Per defocus step: 40 text images (20 Times, 20 Verdana), so 800 text images over the 20 steps.
