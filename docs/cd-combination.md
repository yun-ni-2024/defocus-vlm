# Combining methods (c) and (d)

Method (c) adapts the VLM's vision encoder (a LoRA adapter, slot 2 of the pipeline). Method (d) deconvolves the photograph before the VLM reads it (a learned Wiener filter, slot 1). This document collects everything about using both at once: the variants, their results and what they show. It is updated as new variants are run.

All numbers are on the test set (800 photographs, 40 per blur step), unless marked as validation.

## How the results are read

The training set has no digits; about 14% of the test characters are digits. Mixing the two hides the main effects, so every result is read in two parts:

- **Letters: which method reads better.** Methods are compared on letter accuracy.
- **Digits: does the method generalise to characters it never saw in training.** Every method that contains a LoRA adapter trained on this training set fails here; (d) alone does not.

Letter accuracy and digit accuracy split the evaluation's character accuracy (`src/eval/metrics.py`) into two parts:

1. The ground truth and the prediction, with whitespace removed, are aligned by edit distance (the same alignment the character accuracy counts).
2. Every edit is charged to one class: a substituted or deleted character to the class of the ground-truth character; an inserted character to its own class.
3. Per image and class: accuracy = max(0, 1 - errors of the class / ground-truth characters of the class). Images without digits have no digit accuracy. Per step, the images are averaged, as in the evaluation.

Letter errors plus digit errors equal the edit distance of the evaluation. A digit read as a letter counts only against digits, so it does not lower letter accuracy.

One artefact: under heavy blur the frozen VLM writes coordinate-like strings such as `(181,187),(807,484)`; now and then one of their digits lines up with a ground-truth digit, which gives it a digit accuracy above 0 by chance (0.165 at steps 15 to 19, where its letter accuracy is 0.011).

These numbers are computed from `predictions.csv` of each result folder; the code will move into `src/eval/char_classes.py`.

## E0: both parts as trained, no retraining

Slot 1 is the filter of (d) (`assets/d_text/params.json`, the only (d) version that works); slot 2 is one of the two (c) adapters. Nothing is retrained.

| Config | Slot 1 | Slot 2 |
|---|---|---|
| `configs/e_cd_feature.yaml` | (d) filter | (c) adapter trained with feature distillation (`configs/c_real.yaml`) |
| `configs/e_cd_text.yaml` | (d) filter | (c) adapter trained with the text loss (`configs/c_real_text.yaml`) |

Both test runs: commit `495820d`, clean (`assets/results/e_cd_feature/`, `assets/results/e_cd_text/`).

### Letters

| Test steps | (a) frozen | (c) feature | (c) text | (d) text | E0 feature | E0 text |
|---|---|---|---|---|---|---|
| 0 to 5 | 0.975 | 0.986 | 0.998 | 0.984 | 0.980 | 0.994 |
| 6 to 11 | 0.155 | 0.958 | 0.978 | 0.965 | 0.912 | 0.920 |
| 12 to 14 | 0.032 | 0.904 | 0.932 | 0.926 | 0.775 | 0.813 |
| 15 to 19 | 0.011 | 0.659 | 0.750 | 0.734 | 0.509 | 0.506 |
| mean | 0.347 | 0.884 | 0.920 | 0.907 | 0.811 | 0.823 |

### Digits

| Test steps | (a) frozen | (c) feature | (c) text | (d) text | E0 feature | E0 text |
|---|---|---|---|---|---|---|
| 0 to 5 | 0.993 | 0.752 | 0.687 | 0.998 | 0.760 | 0.657 |
| 6 to 11 | 0.142 | 0.210 | 0.279 | 0.992 | 0.338 | 0.248 |
| 12 to 14 | 0.005 | 0.096 | 0.089 | 0.979 | 0.127 | 0.099 |
| 15 to 19 | 0.165 | 0.025 | 0.034 | 0.817 | 0.021 | 0.020 |
| mean | 0.382 | 0.309 | 0.312 | 0.948 | 0.354 | 0.292 |

For reference, the evaluation's mean character accuracy (letters and digits together): E0 feature 0.747, E0 text 0.748, against 0.802 for (c) feature, 0.834 for (c) text and 0.913 for (d).

### What E0 shows

- **Letters: putting the two parts together makes both worse.** Both E0 versions are below (c) and (d) alone at every step from 6 on, most at heavy blur (about 0.51 against 0.73 to 0.75 at steps 15 to 19). The adapters were trained on blurred photographs and never saw the filter's output, which is sharper but has ringing and amplified noise; what they learned does not carry over to it.
- **Digits: adding an adapter removes the digit reading of (d).** (d) alone reads digits at 0.948; with either adapter in slot 2 it drops to the level of (c) alone (0.29 to 0.35). The filter makes the digits visible, but the adapted encoder no longer reads them.
- On letters, (c) with the text loss and (d) are close (0.920 and 0.907), (c) slightly ahead at every group of steps; whether this is beyond noise is not tested yet.

## Next: E1 and E2

E0 shows that the parts must be trained together to work together. E1 trains the adapter on the filter's output, with the filter fixed; E2 trains the filter through the adapted VLM, with the adapter fixed. Their variants and results will be added here.
