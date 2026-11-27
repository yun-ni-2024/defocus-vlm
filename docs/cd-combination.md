# Combining methods (c) and (d)

Method (c) adapts the VLM's vision encoder (a LoRA adapter, slot 2 of the pipeline). Method (d) deconvolves the photograph before the VLM reads it (a learned Wiener filter, slot 1). This document collects everything about using both at once: the variants, their results and what they show. It is updated as new variants are run.

All numbers are on the test set (800 photographs, 40 per blur step), unless marked as validation.

## How the results are read

The training set has no digits; about 14% of the test characters are digits. Mixing the two hides the main effects, so every result is read in two parts:

- **Letters: which method reads better.** Methods are compared on letter accuracy.
- **Digits: does the method generalise to characters it never saw in training.** Every method that contains a LoRA adapter trained on this training set fails here; (d) alone does not.

Letter accuracy and digit accuracy split the evaluation's character accuracy (`src/eval/metrics.py`) into two parts:

1. The ground truth and the prediction, with whitespace removed, are aligned by edit distance (the same alignment the character accuracy counts).
2. Every edit is charged to one class: a substituted or deleted character to the class of the ground-truth character; an inserted letter or digit to its own class (inserted punctuation to neither).
3. Per image and class: accuracy = max(0, 1 - errors of the class / ground-truth characters of the class). Images without digits have no digit accuracy. Per step, the images are averaged, as in the evaluation.

Letter errors plus digit errors (plus inserted punctuation) equal the edit distance of the evaluation. A digit read as a letter counts only against digits, so it does not lower letter accuracy.

One artefact: under heavy blur the frozen VLM writes coordinate-like strings such as `(181,187),(807,484)`; now and then one of their digits lines up with a ground-truth digit, which gives it a digit accuracy above 0 by chance (0.165 at steps 15 to 19, where its letter accuracy is 0.011).

The tables are produced by `python scripts/char_classes.py --groups 0-5,6-11,12-14,15-19,0-19 <result folders>` (`src/eval/char_classes.py`), from `predictions.csv` of each result folder.

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
| 6 to 11 | 0.156 | 0.959 | 0.978 | 0.966 | 0.912 | 0.920 |
| 12 to 14 | 0.032 | 0.904 | 0.932 | 0.927 | 0.776 | 0.813 |
| 15 to 19 | 0.011 | 0.659 | 0.750 | 0.739 | 0.509 | 0.506 |
| mean | 0.347 | 0.884 | 0.920 | 0.909 | 0.811 | 0.823 |

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

- **Letters: putting the two parts together makes both worse.** Both E0 versions are below (c) and (d) alone at every step from 6 on, most at heavy blur (about 0.51 against 0.74 to 0.75 at steps 15 to 19). The adapters were trained on blurred photographs and never saw the filter's output, which is sharper but has ringing and amplified noise; what they learned does not carry over to it.
- **Digits: adding an adapter removes the digit reading of (d).** (d) alone reads digits at 0.948; with either adapter in slot 2 it drops to the level of (c) alone (0.29 to 0.35). The filter makes the digits visible, but the adapted encoder no longer reads them.
- On letters, (c) with the text loss and (d) are close (0.920 and 0.909), (c) slightly ahead at every group of steps; whether this is beyond noise is not tested yet.

## E1 and E2: training one part for the other

E0 shows that the parts must be trained together to work together. Two ways of doing it, each with both kinds of (c):

| Variant | Fixed | Trained | Loss | Iteration 0 is |
|---|---|---|---|---|
| E1 ctext | (d) filter in slot 1 | a new adapter (starts at zero), on the filter's output | text | (d) alone (filter + original VLM) |
| E1 cfeature | (d) filter in slot 1 | a new adapter (starts at zero), on the filter's output | feature distillation | (d) alone |
| E2 ctext | the (c) text-loss adapter in slot 2 | a new filter, from the ordinary Wiener filter | text | ordinary Wiener + VLM with that adapter |
| E2 cfeature | the (c) feature-distillation adapter in slot 2 | a new filter, from the ordinary Wiener filter | text | ordinary Wiener + VLM with that adapter |

Everything else follows the chosen run of the part being trained: E1 as `train_c_real_text_8k.yaml` or `train_c_real_lr5e-4_8k.yaml`, E2 as `train_d_text_2k.yaml`. The filter is only trained with the text loss, since the pixel and feature losses did not work for (d). Checkpoints are chosen by validation accuracy, as for (c) and (d); each chosen variant is read on the test set once.

| Variant | Training config | Chosen part and records | Evaluation config | Test results |
|---|---|---|---|---|
| E1 ctext | `train_e1_ctext.yaml` | `assets/e1_ctext/` | `e1_ctext.yaml` | `assets/results/e1_ctext/` |
| E1 cfeature | `train_e1_cfeature.yaml` | `assets/e1_cfeature/` | `e1_cfeature.yaml` | `assets/results/e1_cfeature/` |
| E2 ctext | `train_e2_ctext.yaml` | `assets/e2_ctext/` | `e2_ctext.yaml` | `assets/results/e2_ctext/` |
| E2 cfeature | `train_e2_cfeature.yaml` | `assets/e2_cfeature/` | `e2_cfeature.yaml` | `assets/results/e2_cfeature/` |

### Why E1 is expected to work better than E2

This is reasoning written before the results, to be checked against them.

1. **The part that has to change is the reader, and only the reader has the capacity to change.** The mismatch E0 shows lies in the adapter: it has learned what blurred strokes look like. In E2 the filter has to make up for this, but it has 136 parameters and can only set a gain per frequency (one K(f) curve per radius); it can sharpen more or less, not produce an arbitrary image. Its simplest way to please the adapter is to sharpen less, which gives up what (d) gains. In E1 the adapter (10.5 M parameters) adapts instead, and can learn to read through the filter's ringing and amplified noise.
2. **An adapter must see at training the input it gets at test time.** In the pipeline photograph → filter → reader, the downstream part should be trained on the actual output of the upstream part; E0 shows what happens otherwise. E1 does this: the adapter is trained and tested on the filter's output. E2 does not: its adapter was trained on blurred photographs and is tested on the output of a filter, which cannot turn its output back into a blurred photograph.

What E1 does not fix: its adapter is still trained on a set without digits, so it is expected to lose digits like (c). (It mostly did not; see the results.)

## Results of E1 and E2

All four runs: commit `8b6839d`, clean, Runpod RTX PRO 4500 Blackwell, one pod per run.

### Training and checkpoint choice (validation, held-out training pairs, no digits)

| Variant | Validation at iteration 0 | Best validation (iteration) | At the end |
|---|---|---|---|
| E1 ctext | 0.905 | **0.981 (3000)** | 0.976 (8000) |
| E1 cfeature | 0.906 | 0.968 (4000) | 0.962 (8000) |
| E2 ctext | 0.767 | 0.872 (2000) | 0.872 (2000) |
| E2 cfeature | 0.757 | 0.851 (1750) | 0.848 (2000) |

For reference, the chosen checkpoints of the parts alone: (c) text 0.921, (c) feature 0.885, (d) 0.904. Iteration 0 of E1 is (d) alone (0.905 against 0.904 for (d)'s own run); iteration 0 of E2 is the ordinary Wiener filter read with the adapter, which is below the ordinary Wiener filter read by the original VLM (0.831): the adapter makes the filtered image harder to read. E1 is flat from about 2000 iterations on (0.969 to 0.981); E2 rises slowly over its 2000 iterations (E2 cfeature levels off after about 1000).

Test runs of the chosen checkpoints: E2 ctext commit `b52a55b`, E2 cfeature `fe34160`, E1 cfeature and E1 ctext `0255ea1`, all clean (`assets/results/e{1,2}_c{text,feature}/`).

### Letters (test)

| Test steps | (c) feature | (c) text | (d) text | E0 feature | E0 text | E2 cfeature | E2 ctext | E1 cfeature | E1 ctext |
|---|---|---|---|---|---|---|---|---|---|
| 0 to 5 | 0.986 | 0.998 | 0.984 | 0.980 | 0.994 | 0.984 | 0.996 | 0.987 | 0.999 |
| 6 to 11 | 0.959 | 0.978 | 0.966 | 0.912 | 0.920 | 0.926 | 0.939 | 0.984 | 0.995 |
| 12 to 14 | 0.904 | 0.932 | 0.927 | 0.776 | 0.813 | 0.825 | 0.863 | 0.972 | 0.988 |
| 15 to 19 | 0.659 | 0.750 | 0.739 | 0.509 | 0.506 | 0.596 | 0.644 | 0.928 | 0.939 |
| mean | 0.884 | 0.920 | 0.909 | 0.811 | 0.823 | 0.846 | 0.871 | 0.969 | **0.981** |

### Digits (test)

| Test steps | (c) feature | (c) text | (d) text | E0 feature | E0 text | E2 cfeature | E2 ctext | E1 cfeature | E1 ctext |
|---|---|---|---|---|---|---|---|---|---|
| 0 to 5 | 0.752 | 0.687 | 0.998 | 0.760 | 0.657 | 0.767 | 0.667 | 0.965 | 0.968 |
| 6 to 11 | 0.210 | 0.279 | 0.992 | 0.338 | 0.248 | 0.416 | 0.269 | 0.907 | 0.906 |
| 12 to 14 | 0.096 | 0.089 | 0.979 | 0.127 | 0.099 | 0.209 | 0.128 | 0.817 | 0.832 |
| 15 to 19 | 0.025 | 0.034 | 0.817 | 0.021 | 0.020 | 0.047 | 0.067 | 0.566 | 0.562 |
| mean | 0.309 | 0.312 | **0.948** | 0.354 | 0.292 | 0.398 | 0.317 | 0.826 | 0.828 |

For reference, the evaluation's mean character accuracy: E1 ctext 0.960, E1 cfeature 0.949, E2 ctext 0.793, E2 cfeature 0.782, against 0.913 for (d) and 0.834 for (c) text.

### What E1 and E2 show

- **Letters: E1 is the best method of the project.** E1 ctext reaches 0.981 (0.920 for (c) text, 0.909 for (d)); the gain is largest under heavy blur (0.939 against 0.750 and 0.739 at steps 15 to 19). Validation and test agree (0.981 and 0.981). The text loss is again ahead of feature distillation (0.981 against 0.969).
- **E2 recovers part of what E0 lost, but stays below both parts alone** (letters 0.871 and 0.846). The filter can adapt somewhat to the adapted reader, not enough. This matches the reasoning above: the part that has to change is the reader, and the filter has too little freedom to make up for it.
- **Digits: E1 keeps most of the digit reading, against the expectation above.** Its adapter was trained on the same training set without digits, yet it reads digits at 0.83 (0.31 for (c), 0.95 for (d)): nearly as well as (d) on light blur (0.97 at steps 0 to 5), worse under heavy blur (0.56 against 0.82 at steps 15 to 19). E2 keeps the digit failure of its fixed (c) adapter (0.32 to 0.40).

A possible reason for the digit result, not tested: on the filter's output the characters are already nearly readable, so the adapter only has to learn small, character-independent corrections (reading through ringing and noise), which carry over to digits. On the blurred photographs, (c)'s adapter has to learn what each blurred character looks like, which it only learns for the letters it sees.

### What this means

E0, E1 and E2 together show that combining the two parts works only when the reader is trained on the output of the filter. In that order, the physics-based filter does the part it can do (undoing a disc blur whose radius it estimates), the adapter learns to read what remains, and the combination is better than either part on letters and keeps most of the generalisation of (d) to digits.
