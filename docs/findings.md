# Findings

Observations worth keeping for the report and the poster, newest last. Each entry says what was seen and where the evidence is (a run folder under `runs/`, or the exact images and settings).

## 2026-11-08

### The test set contains both fonts

The test set has text images in Verdana (`verdanaRef`) and in Times (`timesR`), for example `focusStep_0_timesR_size_30_sample_0001.tif` and `focusStep_0_verdanaRef_size_30_sample_0018.tif`.

### The frozen VLM invents plausible text on blurred images

In a four-image GPU check, frozen Qwen2-VL-2B read two sharp images at test step 0 perfectly. On two blurred images at test step 7, it did not say that it could not read them. Instead it returned fluent but unrelated words:

| Image (test, `CAM02_blurred`) | Ground truth | Model output | Character accuracy |
|---|---|---|---|
| `focusStep_7_timesR_size_30_sample_0001.tif` | `beZtuqe4du / ZagPvxR6Ys / ThSZ AKYWe` | `Instagram / TagPeopleTo / ThisActing` | 0.17 |
| `focusStep_7_timesR_size_30_sample_0002.tif` | `A7Pbs cMzj / zPRNRieHD2 / 9QmmwqTMdE` | `ATM (My / FEAR is 100 / X000…` (repeated zeros up to the token limit) | 0.00 |

Settings:

- model: `Qwen/Qwen2-VL-2B-Instruct` at commit `895c3a49bc3fa70a340399125c650a463535e71c`, bfloat16, greedy decoding, at most 48 new tokens;
- image: 16-bit TIFF divided by its 99.5th percentile, 8-bit RGB, long side resized to 1024, processor `max_pixels = 448*28*28`;
- prompt: "Transcribe the text in this image exactly as written. Output only the text, one line per line of text.";
- character accuracy: 1 minus the edit distance divided by the length of the ground truth, whitespace ignored.

This was a one-off check, not an experiment: there is no run folder. The baseline experiment (a) will measure this systematically.

## 2026-11-14

### The training set has no digits, the test set does

No transcription in the training set contains a digit; most lines in the test set contain at least one. A model trained only on the real training pairs (method (c), real version) never sees a blurred digit. Each test step has 20 Times and 20 Verdana text images.

## 2026-11-26

### Method (c), real version: how the learning rate and the checkpoint were chosen

Five training runs were compared, all on the same Runpod RTX 4090 and code commit `24a2a86`. They differ only in the setting named in the table; everything else is as in `configs/train_c_real.yaml` (LoRA rank 16, batch 2, 200 warm-up iterations, seed 0).

Each run measured validation accuracy every 500 iterations on the 400 held-out training pairs (sample number a multiple of 10, 20 per step), with the same reading and scoring as the evaluation. The test set was not used.

| Training config | Changed setting | Validation accuracy at 4000 | Best validation accuracy (iteration) |
|---|---|---|---|
| `train_c_real_lr1e-4` | learning rate 1e-4 | 0.776 | 0.776 (4000) |
| `train_c_real` | learning rate 2e-4 | 0.818 | 0.818 (4000) |
| `train_c_real_lr5e-4_8k` | learning rate 5e-4, 8000 iterations | 0.843 | **0.885 (7000)** |
| `train_c_real_lr1e-3` | learning rate 1e-3 | 0.804 | 0.824 (3500) |
| `train_c_real_lr5e-4_r32` | learning rate 5e-4, LoRA rank 32 (alpha 64) | 0.839 | 0.842 (3500) |

Before training (iteration 0) every run scores 0.346. The 8000-iteration run also serves as the 4000-iteration run at learning rate 5e-4.

What the table shows:

- 5e-4 is the best learning rate. At 1e-4 and 2e-4 the curve is still rising at 4000 iterations; at 1e-3 it goes up and down after 2000.
- Rank 32 is no better than rank 16.
- At 5e-4, running to 8000 iterations adds about 0.04; after about 6500 the curve is flat apart from noise (0.873, 0.885, 0.865, 0.874 at 6500 to 8000).

Rule: the checkpoint with the highest validation accuracy is used. This is iteration 7000 of `train_c_real_lr5e-4_8k`, saved in `assets/c_real/adapter/` and set as `model.lora` in `configs/c_real.yaml`. The test set was then read once with this adapter: mean character accuracy 0.802 over the 800 images (`assets/results/c_real/`).

Evidence: `assets/c_real/training/<training config>/log.csv` (training loss and validation accuracy of each run).

### Method (d): which training loss works, how the checkpoint was chosen, and the test result

All runs on Runpod RTX PRO 4500 Blackwell (no RTX 4090 was free in the volume's data centre). Every run starts from the ordinary Wiener filter with the estimated radius (K = 3e-3), which scores 0.831 on the validation pairs (the same 400 held-out training pairs as method (c)). The test set was not used for any choice.

| Training config | Loss | Learning rate (log K) | Steps | Best validation accuracy (step) | At the end |
|---|---|---|---|---|---|
| `train_d_pixel` | pixel MSE | 1e-2 | 2000 | 0.831 (0) | 0.416 |
| `train_d_feature` | visual-token MSE | 1e-2 | 2000 | 0.831 (0) | 0.805 |
| `train_d_feature_lr3e-3` | visual-token MSE | 3e-3 | 2000 | 0.831 (0) | 0.821 |
| `train_d_text` | text cross-entropy | 1e-2 | 500 | 0.897 (500) | 0.897 |
| `train_d_text_2k` | text cross-entropy | 1e-2 | 2000 | **0.904 (1500)** | 0.903 |

What the table shows:

- Both MSE losses decrease during training while accuracy falls, on the validation pairs and equally on a training subset (400 pairs: 0.828 to 0.804 for the feature loss, 0.828 to 0.418 for the pixel loss). With 136 parameters this is not overfitting: the losses disagree with reading accuracy. A loss audit on fixed filters (`scripts/audit_losses.py`) confirms it without training: only the text loss ranks the three filters as the accuracy does.
- The text loss raises accuracy from the first steps; after about 500 steps the curve is flat (0.896 to 0.904) and does not fall.
- The gain is at heavy blur: on validation, step 19 goes from 0.11 to 0.57 and step 18 from 0.31 to 0.63; steps 0 to 5 are unchanged.

Rule: the checkpoint with the highest validation accuracy of the text-loss runs is used. This is step 1500 of `train_d_text_2k`, saved as `assets/d_text/params.json` and set in `configs/d_text.yaml`. The test set was then read once with it (commit `e88898f`): mean character accuracy **0.913** over the 800 images (`assets/results/d_text/`), against 0.352 for method (a) and 0.802 for method (c), real version. Per step it is at least as high as method (c) everywhere, and 0.57 against 0.37 at step 19.

Validation to test: method (d) 0.904 to 0.913, method (c) 0.885 to 0.802. A possible reason, not tested yet: the test set contains digits and the training set does not (entry of 2026-11-14); the adapted encoder of method (c) can learn what letters look like, the 136-parameter filter cannot.

Evidence: `assets/d_text/training/<training config>/log.csv` for the text-loss runs; the pixel and feature runs and the audit table are in `runs/train_d_pixel_20261126-211036`, `runs/train_d_feature_20261126-202722`, `runs/train_d_feature_lr3e-3_20261126-215036` and `runs/audit_losses_20261126-230543.csv` under `DEFOCUS_ROOT` on the Runpod volume. The text-loss runs record commit `6e3b54b` with `dirty: true`; their code is that of commit `3485423` (checked by sha256). Reasoning and the full audit: `docs/d-loss-design.md`.

## 2026-11-27

### Method (c), real version, with the text loss: learning rate, checkpoint and test result

Method (c) trained with the text loss of method (d) (`src/vlm/text_loss.py`, `loss: text`) instead of feature distillation: the loss is the cross-entropy of the correct transcription given the blurred photograph, so no sharp photograph is needed. Everything else is as in `configs/train_c_real.yaml` (LoRA rank 16, batch 2, 200 warm-up iterations, seed 0). All runs on Runpod RTX PRO 4500 Blackwell; validation on the same 400 held-out training pairs as before; the test set was not used for any choice.

| Training config | Learning rate | Iterations | Best validation accuracy (iteration) | At the end |
|---|---|---|---|---|
| `train_c_real_text_lr1e-4` | 1e-4 | 3000 | 0.863 (3000) | 0.863 |
| `train_c_real_text` | 3e-4 | 3000 | 0.879 (3000) | 0.879 |
| `train_c_real_text_lr1e-3` | 1e-3 | stopped at 1000 | 0.596 (500) | 0.020 |
| `train_c_real_text_8k` | 3e-4 | 8000 | **0.921 (6000)** | 0.895 |

What the table shows:

- 3e-4 is the best learning rate; at 1e-3 the training diverges (loss about 5, accuracy 0.020 at 1000), so the run was stopped.
- At 3e-4 the text loss reaches 0.886 at 3500 iterations, the best value of feature distillation (0.885 at 7000). From 4000 on the curve is flat apart from noise (0.893 to 0.921).
- The 8000-iteration run has the same length and schedule as the chosen feature-distillation run, so the two differ only in the loss.
- On validation the text loss is better at every step, most at heavy blur: step 18 0.64 against 0.50, step 19 0.57 against 0.40; steps 0 to 5 reach 0.99 to 1.00.

Rule: the checkpoint with the highest validation accuracy is used. This is iteration 6000 of `train_c_real_text_8k`, saved in `assets/c_real_text/adapter/` and set in `configs/c_real_text.yaml`. The test set was then read once with it (commit `32f641d`, clean): mean character accuracy **0.834** over the 800 images (`assets/results/c_real_text/`).

| Test steps | (a) frozen | (c) feature | (c) text | (d) text |
|---|---|---|---|---|
| 0 to 5 | 0.978 | 0.956 | 0.957 | 0.985 |
| 6 to 11 | 0.154 | 0.847 | 0.875 | 0.969 |
| 12 to 14 | 0.028 | 0.793 | 0.816 | 0.933 |
| 15 to 19 | 0.031 | 0.569 | 0.648 | 0.746 |
| mean | 0.352 | 0.802 | 0.834 | 0.913 |

Validation to test: 0.921 to 0.834, the same drop as with feature distillation (0.885 to 0.802). The loss does not remove the drop; the next entry shows where it comes from.

Evidence: `assets/c_real_text/training/<training config>/log.csv`. The three 3000-iteration runs record commit `35a5ff3`, the 8000-iteration run `32f641d`, all clean. The reason the 1e-3 run was stopped is in `logs/queue_c_text.log` under `DEFOCUS_ROOT` on the Runpod volume.

### Method (c) loses digits; method (d) does not

This tests the possible reason given in the method (d) entry. About 14% of the test characters are digits; the training set has none (entry of 2026-11-14). Each ground-truth character was counted as read or not, separately for digits and letters:

| Test steps | Characters | (a) frozen | (c) feature | (c) text | (d) text |
|---|---|---|---|---|---|
| 0 to 4 | digits | 1.000 | 0.843 | 0.759 | 0.999 |
| 0 to 4 | letters | 0.988 | 0.989 | 0.998 | 0.988 |
| 5 to 14 | digits | 0.210 | 0.196 | 0.241 | 0.993 |
| 5 to 14 | letters | 0.269 | 0.950 | 0.968 | 0.958 |
| 15 to 19 | digits | 0.225 | 0.023 | 0.034 | 0.851 |
| 15 to 19 | letters | 0.043 | 0.672 | 0.765 | 0.752 |

What the table shows:

- Both versions of method (c) almost stop reading digits under blur, and even on nearly sharp photographs (steps 0 to 4) they read fewer digits than the frozen model. The text loss loses more digits there than feature distillation (0.759 against 0.843).
- On letters, method (c) with the text loss is as good as method (d) or better (0.968 against 0.958 at steps 5 to 14, 0.765 against 0.752 at steps 15 to 19). Almost all of the test gap between (c) and (d) comes from digits.
- Method (d) changes only the image (136 filter parameters) and leaves the VLM as it is, so it reads digits as well as letters. The adapted encoder of method (c) learns what the training characters look like and does not carry over to characters it never saw.

How it was counted: `python scripts/char_classes.py --groups 0-4,5-14,15-19 assets/results/baseline_a assets/results/c_real assets/results/c_real_text assets/results/d_text` (`src/eval/char_classes.py`). It reads `predictions.csv` of each result folder and removes whitespace from the ground truth and the prediction; a ground-truth character counts as read if it lies in a matching block of Python's `difflib.SequenceMatcher(None, truth, prediction, autojunk=False)`. This is a different measure from the character accuracy of the evaluation (edit distance), so the numbers are not directly comparable with the tables above.
