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
