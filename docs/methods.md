# Methods

The four methods compared in the proposal (`CSC2529 — Proposal.pdf`). All of them use the same frozen language model of Qwen2-VL-2B, and at test time none of them is told the defocus step or the PSF of a photograph.

| | Method | What reaches the VLM | What is trained |
|---|---|---|---|
| (a) | Baseline | the blurred photograph | nothing |
| (b) | Deconvolution | the photograph deconvolved with an estimated PSF | nothing |
| (c) | VLM adaptation | the blurred photograph | LoRA on the vision encoder |
| (d) | Learnable deconvolution | the output of a trained deconvolution module | the module's regularisation |

## (a) Baseline

The blurred photograph goes directly to the VLM.

## (b) Deconvolution with an estimated PSF

1. Model the PSF as a disc, so only its radius has to be estimated.
2. Estimate the radius from the photograph itself.
3. Deconvolve the photograph with that disc and give the result to the VLM.

Deconvolution methods: Wiener filtering and ADMM with a TV prior (from the course), plug-and-play with a learned denoiser (DPIR), and the Reconstruct Anything Model (RAM).

## (c) VLM adaptation

LoRA adapters on the vision encoder, trained with feature distillation: the features of a blurred image should match the features the frozen encoder gives for its sharp version (as in MAA and Uni-DocRobust). At test time the adapted VLM reads the blurred photograph directly. Two versions:

1. trained on the real sharp/blurred pairs of the HDC2021 training set;
2. trained on simulated pairs: rendered text blurred with PSFs measured from the calibration targets.

## (d) Learnable deconvolution

A Wiener filter with learned regularisation. It takes the blurred photograph and the PSF estimated as in (b); because it is trained, it can learn to compensate for errors in that estimate. Two versions:

1. VLM-feature loss: the difference between the frozen encoder's features of the module's output and of the sharp image;
2. pixel loss: the difference between the module's output and the sharp image.

## Reference: calibrated PSF

Deconvolution with the PSF measured from the calibration targets of the right step. It uses information no test-time method has, so it serves only as an upper bound for (b).
