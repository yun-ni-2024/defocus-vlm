# References

The papers cited in the proposal, grouped by what they are used for.

## Data and model

- **Juvonen et al., 2023.** Helsinki Deblur Challenge 2021: the dataset (`dataset.md`).
- **Trippe et al., 2023.** One of the challenge entries; entries were scored with the Tesseract OCR engine.
- **Wang et al., 2024.** Qwen2-VL, the VLM we use.

## Motivation

- **He et al., NeurIPS 2025.** "Seeing is Believing?": VLMs produce plausible text that is not in a degraded image (OCR hallucination).

## Deconvolution, method (b)

- **Zhang et al., 2021.** DPIR: plug-and-play restoration with a learned denoiser prior.
- **Terris et al., 2026.** Reconstruct Anything Model (RAM): a pretrained network that takes the forward operator as an input.

## VLM adaptation, method (c)

- **Xing et al., 2025.** Restoration in front of a VLM does not reliably help on generic corruptions.
- **Li et al., 2026.** MAA: trains the vision side so that features of a degraded image match those of the clean image.
- **Zhou et al., 2026.** Uni-DocRobust: the same idea for document images.

## Learnable deconvolution, method (d)

- **Liu et al., 2018.** A denoiser trained with the loss of a frozen classifier.
- **Diamond et al., 2021.** "Dirty Pixels": a whole physics-structured pipeline trained for classification; recognition improves while PSNR drops.
- **Sitzmann et al., 2018.** End-to-end optics design with a differentiable Wiener-type deconvolution layer.

Full entries: `refs.bib` of the proposal.
