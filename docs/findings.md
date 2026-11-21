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
