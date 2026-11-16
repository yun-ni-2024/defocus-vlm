# Pilot results (October 2026)

Before the proposal, we ran a small feasibility test on a laptop CPU, on real HDC2021 photographs from the training set. Each point used only 4 to 10 images, so these numbers show what to expect, not results. The pilot code is not in this repository.

One difference from the final setup: the pilot's deconvolution used a disc PSF whose radius was measured from the calibration targets of the right step, i.e. the PSF was known. In the project, (b) must estimate it from the photograph; the pilot numbers are closer to the calibrated-PSF upper bound.

## Frozen Qwen2-VL-2B, method (a)

| Step | Character accuracy |
|---|---|
| 5 | 96% |
| 7 | 26% |
| 8 and above | below 10% |

When it fails, it writes invented text rather than saying it cannot read the image (see also `findings.md`).

## Wiener deconvolution with the calibrated disc, then the VLM

- 91 to 98% at steps 7 to 14, 78% at step 16, 67% at step 17, almost nothing at step 19 (plug-and-play also fails there).
- A radius 13 to 17% too small loses most of the gain, so (b) depends on a good radius estimate.
- With the same Wiener outputs, the Tesseract OCR engine reads up to about step 9 and the VLM up to step 16: much of the gap to the 2021 challenge results comes from the reader, not the deblurring.

## LoRA with feature distillation, method (c)

150 training iterations on simulated blur only:

| Real test image | Before | After |
|---|---|---|
| step 7 | 0.22 | 0.93 |
| step 8 | 0.07 | 0.73 |
| sharp | unchanged | unchanged |

Its remaining errors are well-formed wrong letters, which are hard to notice.

## Not run in the pilot

The trained deconvolution module (d), the pixel-loss comparison, and training (c) on real pairs.
