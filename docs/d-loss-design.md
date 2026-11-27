# Choosing the training loss of method (d)

Method (d) trains a small Wiener filter (136 parameters, `src/frontend/wiener.py`) in front of the frozen
Qwen2-VL-2B. The filter is trained by gradient descent through the VLM, so it needs a differentiable
loss. This note records why the first two losses failed, how the text loss was chosen, and the
evidence for each step. All numbers are on the validation split (400 photographs, all 20 steps)
unless stated otherwise.

## 1. The goal and why it cannot be optimised directly

What we want is the reading accuracy of the whole pipeline:

$$
\max_\theta\ \mathbb{E}_{(y,t)}\Big[A\big(\mathrm{decode}(V(R_\theta(y))),\ t\big)\Big]
$$

with `y` the blurred photograph, `t` the correct text, `R_θ` the filter, `V` the VLM and `A` the
character accuracy. Greedy decoding takes an argmax at every position, so a small change of θ
usually leaves the output text unchanged: the accuracy is piecewise constant in θ and its gradient is
zero almost everywhere. Training therefore needs a differentiable stand-in (a surrogate loss) whose
decrease means an increase of accuracy.

## 2. First attempt: two MSE losses

| Loss | Definition |
|---|---|
| pixel | MSE between the filtered image and the sharp photograph (CAM01) at the size the VLM sees, after a least-squares gain and offset per image |
| feature | MSE between the visual tokens of the filtered image and those of the sharp photograph (the loss of method (c)) |

Both runs started from the ordinary Wiener filter (K = 3e-3, estimated radius), validation accuracy 0.831.

| Run | Training loss (mean of first 250 steps → last 250 steps) | Validation accuracy (start → step 2000) |
|---|---|---|
| `train_d_feature` (lr 1e-2) | 1.55 → 1.39 | 0.831 → 0.805 |
| `train_d_feature_lr3e-3` | 1.59 → 1.45 | 0.831 → 0.821 |
| `train_d_pixel` (lr 1e-2) | 0.030 → 0.026 | 0.831 → 0.416 |

The optimiser did its job (the training loss went down), but accuracy went down too. On a training
subset (400 photographs, 20 per step) the accuracy fell in the same way: 0.828 → 0.804 (feature) and
0.828 → 0.418 (pixel). The filter has only 136 parameters and both splits come from the same
distribution, so this is not overfitting: the losses do not measure what the evaluation measures.
A larger learning rate only made the feature run worse faster (0.794 after 250 steps at lr 3e-2).

What the filter learned:
- pixel: K at middle frequencies rose about 100 times (more smoothing) and the radius correction of
  the two largest radius bins reached its limit of +22 %. The images became blurry; accuracy collapsed
  from step 6 on.
- feature: no systematic drift, but the radius correction of the largest bin (steps 18–19) moved to
  0.93; the losses were concentrated there (step 18: 0.31 → 0.07).

## 3. Why MSE losses push towards blur

For any squared-error loss the best output is the mean of all sharp images that could have produced
the blurred one:

$$
\arg\min_{\hat x}\ \mathbb{E}\big[\|\hat x - x\|^2 \,\big|\, y\big] = \mathbb{E}[x \mid y]
$$

For text this distribution has several modes (a blurred letter may be a `c` or an `e`); their mean is
both letters on top of each other, which nobody can read. The stronger the blur, the wider the
distribution and the blurrier the mean. A Wiener filter cannot produce the exact mean, but within its
family the MSE picks the member closest to it: the smoother one, with a larger K. This is the
regression-to-the-mean effect behind the perception–distortion trade-off (Blau and Michaeli 2018).

The feature loss is the same with `x` replaced by the encoder's tokens. In addition, its distance is
dominated by a few outlier channels (largest channel variance 331, median 1.7, measured in
`src/frontend/checks.py`), which need not carry what the letters are.

## 4. What a surrogate loss needs

1. It is measured where the evaluation is measured, on the text the VLM reads, not on an
   intermediate image or feature whose distance is assumed to match readability.
2. It does not prefer the average of the possible answers.
3. It moves with the accuracy: ideally it is an upper bound on the error.

## 5. The text loss

The cross-entropy of the correct transcription under the frozen VLM, with teacher forcing
(`src/frontend/text_loss.py`):

$$
\mathcal{L}_{\text{text}} = -\frac{1}{T}\sum_{k=1}^{T}\log p\big(t_k \,\big|\, t_{<k},\ \text{prompt},\ R_\theta(y)\big)
$$

- It is measured on the text, so requirement 1 holds.
- A blurry image makes the model unsure of every letter, which raises the loss; the image that
  minimises it is the one the VLM reads most confidently. No pull towards the average (requirement 2).
- If token `k` is predicted wrongly, another token has at least its probability, so the correct token
  has probability at most 1/2. Hence

$$
\mathbf{1}\big[\arg\max_v p(v\mid\cdot) \ne t_k\big] \;\le\; -\log_2 p(t_k\mid\cdot)
$$

  and the teacher-forced token error rate is at most the mean cross-entropy in bits (requirement 3).
  This is why classifiers are trained with cross-entropy and evaluated with accuracy
  (classification-calibrated surrogates, Bartlett, Jordan and McAuliffe 2006).

Remaining differences from the evaluation: teacher forcing instead of greedy decoding, tokens instead
of characters, and whitespace (ignored by the metric).

## 6. Checking the losses before training (loss audit)

`scripts/audit_losses.py` restores every validation photograph with three saved filters (the start,
the end of the feature run, the end of the pixel run) and computes all three losses per image.

Mean per filter (accuracy: higher is better; losses: lower is better):

| Filter | Accuracy | pixel | feature | text |
|---|---|---|---|---|
| start (ordinary Wiener) | **0.831** | 0.0293 | 1.532 | **0.939** |
| feature run, step 2000 | 0.805 | 0.0296 | **1.392** | 1.182 |
| pixel run, step 2000 | 0.416 | **0.0267** | 2.295 | 3.863 |

The pixel loss ranks the worst filter best and the feature loss prefers its own (worse) filter; only
the text loss orders the filters as the accuracy does.

For the images whose accuracy changed between the start and a trained filter, the fraction where the
loss moved in the direction agreeing with the accuracy:

| Change | pixel | feature | text |
|---|---|---|---|
| start → feature run (227 images) | 0.59 | **0.41** | 0.71 |
| start → pixel run (296 images) | **0.04** | 0.95 | 0.96 |

Each MSE loss disagrees with the accuracy on the very changes it produced. Spearman correlation of
the per-image loss with the per-image accuracy (all filters together): pixel −0.52, feature −0.82,
text −0.89.

## 7. Training with the text loss

`configs/train_d_text.yaml` and `configs/train_d_text_2k.yaml`: same filter, start and optimiser as
before; 8 pairs per step (4 batches of 2 accumulated); lr 1e-2 for log K and 1e-3 for the radius
correction; 500 and 2000 steps, same seed.

| Step | 0 | 100 | 200 | 250 | 300 | 400 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 2000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 500-step run | 0.831 | 0.869 | 0.880 | | 0.889 | 0.892 | 0.897 | | | | | | |
| 2000-step run | 0.831 | | | 0.884 | | | 0.896 | 0.898 | 0.899 | 0.900 | **0.904** | 0.900 | 0.903 |

The training loss and the accuracy now move together (mean cross-entropy per 100 steps: 1.24, 0.77,
0.59, 0.44 over the first 400 steps). Most of the gain comes in the first 500 steps; after that the
curve is flat within noise and does not fall. The checkpoint with the highest validation accuracy of
both runs is step 1500 of the 2000-step run (0.904), copied to `assets/d_text/params.json`.

Per defocus step (validation):

| Step | 0–5 | 6–12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
|---|---|---|---|---|---|---|---|---|---|
| start | 0.96–0.99 | 0.88–0.96 | 0.82 | 0.86 | 0.83 | 0.75 | 0.58 | 0.31 | 0.11 |
| text loss, step 1500 | 0.96–0.99 | 0.91–0.97 | 0.91 | 0.92 | 0.91 | 0.85 | 0.76 | 0.63 | 0.57 |

The gain is where the start was weakest; light blur is unchanged. What the filter learned is the
opposite of the pixel run: in the largest radius bins K fell from 3e-3 to as low as about 4e-4 at
middle and high frequencies (sharper deconvolution, more noise and ringing) and rose only near the
lowest frequencies; the radius correction stayed between 0.99 and 1.06. The VLM tolerates noise and
ringing much better than blur, which the MSE losses cannot express.

Test split, read once with this filter (`configs/d_text.yaml`, results in `assets/results/d_text/`):
mean character accuracy **0.913** over 800 images, against 0.352 for the blurred photographs read
directly (method (a)) and 0.802 for method (c), real version. Unlike method (c) (validation 0.885,
test 0.802), the filter loses nothing from validation to test (0.904, 0.913).

## 8. Related work

- **Task-based image quality.** Medical imaging judges an imaging system by how well an observer
  (a radiologist, an algorithm, or the ideal Bayesian observer) performs a task with its images, not
  by pixel error. Here the observer is the VLM and the task is reading.
  H. H. Barrett and K. J. Myers, *Foundations of Image Science*, Wiley, 2004.
- **End-to-end image processing for a downstream network.** Training the image-processing pipeline
  (demosaicking, denoising, deblurring) with the classification loss of the network that uses the
  images, instead of for perceptual quality; the learned processing keeps more fine detail at the cost
  of more noise, as found here.
  S. Diamond, V. Sitzmann, F. Julca-Aguilar, S. Boyd, G. Wetzstein, F. Heide, "Dirty Pixels: Towards
  End-to-End Image Processing and Perception", ACM Transactions on Graphics 40(3), 2021.
- **Regression to the mean.** Y. Blau and T. Michaeli, "The Perception-Distortion Tradeoff", CVPR 2018.
- **Cross-entropy as a surrogate for accuracy.** P. L. Bartlett, M. I. Jordan, J. D. McAuliffe,
  "Convexity, Classification, and Risk Bounds", Journal of the American Statistical Association, 2006.
- **If teacher forcing becomes the limit:** sequence-level training on the metric itself, e.g.
  S. J. Rennie et al., "Self-Critical Sequence Training for Image Captioning", CVPR 2017.
