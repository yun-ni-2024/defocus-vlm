# Glossary

Imaging terms used in this project, in plain words.

- **Defocus.** The lens is focused at a different distance than the object, so each point of the object spreads into a small blob on the sensor.
- **PSF (point spread function).** The image of a single point of light. For defocus it is roughly a disc whose radius grows with the amount of defocus.
- **Convolution model.** A blurred photograph is the sharp image convolved with the PSF, plus noise: y = k * x + n.
- **Defocus step.** The HDC2021 setting for how far the blurred camera is out of focus, 0 to 19.
- **LSF (line spread function).** The image of a thin line; it shows the PSF's profile in one direction and is used to measure the PSF's size.
- **Deconvolution.** Recovering x from y when k is known. It is an ill-posed inverse problem: frequencies that the PSF removes cannot be recovered, and dividing by small values amplifies noise.
- **Blind deconvolution.** Deconvolution when k must also be estimated, which is our setting at test time.
- **Wiener filter.** Deconvolution in the Fourier domain that divides by the PSF but adds a regularisation term so that frequencies with little signal are not blown up.
- **ADMM-TV.** Deconvolution posed as an optimisation problem with a total-variation prior (images are mostly flat with sharp edges), solved by the ADMM algorithm.
- **Plug-and-play (PnP).** An optimisation algorithm like ADMM in which the prior step is replaced by a pretrained denoising network.
- **RAM (Reconstruct Anything Model).** A pretrained network that takes the degraded image and the forward operator (here the PSF) and outputs a restored image.
- **LoRA (low-rank adaptation).** Fine-tuning that freezes a weight matrix W and learns a low-rank update BA added to it, so only a few parameters are trained.
- **Feature distillation.** Training a model so that its features on a degraded input match the features a frozen reference model gives on the clean input.
- **Character accuracy.** 1 minus edit distance divided by the length of the ground truth (`evaluation.md`).
- **Hallucination.** The VLM outputs fluent text that is not in the image instead of saying it cannot read it.
