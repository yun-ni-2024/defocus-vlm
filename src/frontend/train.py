"""Training loop for method (d): the learnable Wiener filter, trained through the frozen VLM.

Three losses, selected by the config's `loss`:
  feature: mean squared difference between the visual tokens of the frozen encoder reading the
           filter's output and reading the sharp photograph (the loss of method (c)).
  pixel:   mean squared difference between the filter's output and the sharp photograph at the
           size the VLM sees, after fitting a gain and an offset per image (the two cameras differ
           in brightness and contrast).
  text:    cross-entropy of the correct transcription under the frozen VLM reading the filter's
           output (src/vlm/text_loss.py).
Everything else is the same for both, so a difference between them comes from the loss alone.

An optional `model.lora` (slot 2) loads a fixed (c) adapter into the VLM, which then stays frozen:
E2 trains the filter for the adapted VLM instead of the original one.
"""
import csv
import time

import torch
import torch.nn.functional as F

from src.data.image import load_tif, to_pil
from src.eval.metrics import char_accuracy
from src.frontend.pairs import cache_key, endless_batches, load_radius_cache, split_pairs
from src.vlm.text_loss import TextLoss
from src.vlm.vlm_input import pixel_values, to_vlm_size, visual_tokens
from src.frontend.wiener import build_from_config, restore_numpy
from src.utils.run import defocus_root
from src.vlm.qwen import QwenReader


def feature_loss(encoder, image_processor, size, dtype, restored, sharp):
    """Feature-distillation loss: visual tokens of the restored images against those of the sharp ones.

    Args:
        encoder: the frozen vision encoder.
        image_processor: QwenReader.proc.image_processor.
        size: pixel limits (QwenReader.size).
        dtype: dtype of the encoder's weights.
        restored: tensor (B, h, w), the filter's output after to_vlm_size().
        sharp: tensor (B, h, w), the sharp photographs after to_vlm_size().

    Returns:
        Scalar tensor: mean squared difference over all tokens and channels.
    """
    # teacher: the sharp photographs, no gradients
    with torch.no_grad():
        target = visual_tokens(encoder, *pixel_values(sharp, image_processor, size), dtype)
    # student: the filter's output; gradients flow back through the encoder to the filter
    pred = visual_tokens(encoder, *pixel_values(restored, image_processor, size), dtype)
    # compare in float32 so the loss is not rounded to bfloat16
    return F.mse_loss(pred.float(), target.float())


def pixel_loss(restored, sharp):
    """Pixel loss: mean squared difference after the best gain and offset per image.

    For each image the gain a and offset b minimising |a * restored + b - sharp|^2 are found in
    closed form (least squares), so only differences a linear brightness change cannot explain count.

    Args:
        restored: tensor (B, h, w), the filter's output after to_vlm_size().
        sharp: tensor (B, h, w), the sharp photographs after to_vlm_size().

    Returns:
        Scalar tensor: mean squared difference over all pixels of the batch.
    """
    u = restored.flatten(1)
    v = sharp.flatten(1)
    # least-squares gain and offset per image
    du = u - u.mean(1, keepdim=True)
    dv = v - v.mean(1, keepdim=True)
    a = (du * dv).mean(1, keepdim=True) / du.pow(2).mean(1, keepdim=True).clamp(min=1e-12)
    b = v.mean(1, keepdim=True) - a * u.mean(1, keepdim=True)
    return F.mse_loss(a * u + b, v)


def validate(reader, model, samples, radii, image_cfg, batch_size):
    """Read the validation photographs through the current filter and score them.

    Uses the evaluation path: NumPy image, filter, to_pil(), QwenReader.read().

    Args:
        reader: QwenReader, with the original encoder or a fixed adapter (model.lora).
        model: LearnedWiener being trained.
        samples: list of Sample held out for validation.
        radii: dict from cache_key() to estimated radius.
        image_cfg: the config's image section (white_percentile, long_side).
        batch_size: images per VLM call.

    Returns:
        List of dicts, one per image: step, font, number, radius, truth, prediction, accuracy.
    """
    rows = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        images = []
        for s in batch:
            # slot 1 of the evaluation pipeline, with the filter as it is now
            a = load_tif(s.blurred, image_cfg["white_percentile"])
            images.append(to_pil(restore_numpy(model, a, radii[cache_key(s.blurred)]), image_cfg["long_side"]))
        preds = reader.read(images)
        # score with the same metric as the evaluation (whitespace ignored)
        for s, p in zip(batch, preds):
            rows.append({"step": s.step, "font": s.font, "number": s.number,
                         "radius": round(radii[cache_key(s.blurred)], 2), "truth": s.text,
                         "prediction": p, "accuracy": round(char_accuracy(p, s.text), 4)})
    return rows


def write_csv(path, rows):
    """Write a list of dicts as a CSV file with one column per key.

    Args:
        path: output file.
        rows: list of dicts with the same keys.
    """
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def warmup_factor(it, warmup):
    """Learning-rate multiplier at an iteration: linear warm-up, then 1.

    Args:
        it: iteration, starting at 1.
        warmup: number of warm-up iterations; 0 means none.

    Returns:
        Float in (0, 1].
    """
    return min(1.0, it / warmup) if warmup else 1.0


def train(cfg, run):
    """Train the learnable Wiener filter as the config says and write everything into the run folder.

    Writes into `run`:
      log.csv: iteration, training loss, learning-rate multiplier, validation accuracy, seconds
      val/it<N>.csv: the validation predictions at iteration N (N = 0 is the ordinary Wiener filter)
      params/it<N>.json: the filter at iteration N, loadable by src/restore/learned_wiener.py

    Args:
        cfg: the training config (see configs/train_d_feature.yaml).
        run: the run folder created by new_run().
    """
    m, d, t, v = cfg["model"], cfg["data"], cfg["train"], cfg["validation"]
    torch.manual_seed(t["seed"])
    if cfg["loss"] not in ("feature", "pixel", "text"):
        raise ValueError(f"unknown loss {cfg['loss']!r}")

    # the frozen model, loaded through the class the evaluation uses; no parameter needs gradients.
    # model.lora (optional, slot 2) merges a fixed (c) adapter into the encoder: E2 trains the filter
    # for the adapted VLM
    lora = defocus_root() / m["lora"] if m.get("lora") else None
    reader = QwenReader(defocus_root() / m["path"], dtype=m["dtype"], min_pixels=m["min_pixels"],
                        max_pixels=m["max_pixels"], max_new_tokens=m["max_new_tokens"],
                        prompt=m["prompt"], device=m.get("device", "cuda"), lora=lora)
    for p in reader.model.parameters():
        p.requires_grad_(False)
    encoder = reader.model.model.visual
    dev, dtype = reader.device, reader.model.dtype

    # the filter, starting as the ordinary Wiener filter; float32 for the FFT
    model = build_from_config(cfg["wiener"]).to(dev)
    print(f"trainable parameters: {sum(p.numel() for p in model.parameters())}", flush=True)
    # Adam with a smaller step for the radius correction, to which the result is most sensitive;
    # no weight decay, since pulling log K towards 0 (K = 1) is not a neutral direction
    opt = torch.optim.Adam([{"params": [model.log_k], "lr": t["lr_log_k"]},
                            {"params": [model.log_scale], "lr": t["lr_log_scale"]}], weight_decay=0.0)
    base_lr = [g["lr"] for g in opt.param_groups]

    # training and validation pairs, and the cached estimated radii
    train_set, val_set = split_pairs(d.get("steps"), d["val_every"])
    estimator, radii = load_radius_cache(d["radius_cache"])
    if estimator != d["estimator"]:
        raise ValueError(f"radius cache made by {estimator}, config asks for {d['estimator']}")
    print(f"{len(train_set)} training pairs, {len(val_set)} validation pairs, radii from {estimator}", flush=True)
    batches = endless_batches(train_set, radii, cfg["image"]["white_percentile"], t["batch_size"],
                              t["seed"], t.get("workers", 0))
    # the text loss prepares its token layout once, for photographs of the training size
    text_loss = None
    if cfg["loss"] == "text":
        full_shape = load_tif(train_set[0].blurred).shape
        text_loss = TextLoss(reader, full_shape, cfg["image"]["long_side"], t.get("ce_cap"))
    # batches whose gradients are summed before each step (a larger batch at the same memory)
    accumulate = t.get("accumulate", 1)

    (run / "val").mkdir()
    (run / "params").mkdir()
    log_file = open(run / "log.csv", "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["iteration", "loss", "lr_factor", "val_accuracy", "seconds"])
    t0 = time.time()

    def checkpoint(it, loss, lr_factor):
        """Validate, save the predictions and the filter, and log one line.

        Args:
            it: current iteration (0 = before training).
            loss: training loss of this iteration, or None before training.
            lr_factor: current learning-rate multiplier, or None before training.
        """
        rows = validate(reader, model, val_set, radii, cfg["image"], v["batch_size"])
        write_csv(run / "val" / f"it{it}.csv", rows)
        model.save(run / "params" / f"it{it}.json")
        acc = sum(r["accuracy"] for r in rows) / len(rows)
        log.writerow([it, "" if loss is None else round(loss, 8), "" if lr_factor is None else lr_factor,
                      round(acc, 4), round(time.time() - t0)])
        log_file.flush()
        print(f"iteration {it}: validation accuracy {acc:.3f}", flush=True)

    # starting point: the ordinary Wiener filter with the estimated radius
    checkpoint(0, None, None)

    for it in range(1, t["iterations"] + 1):
        # learning-rate warm-up, applied to both parameter groups
        lr_factor = warmup_factor(it, t["warmup"])
        for g, lr in zip(opt.param_groups, base_lr):
            g["lr"] = lr * lr_factor
        # gradients of `accumulate` batches are summed (each scaled by 1 / accumulate)
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(accumulate):
            # one batch of full-resolution pairs through the filter
            sharp, blurred, r_hat, texts = next(batches)
            sharp, blurred, r_hat = sharp.to(dev), blurred.to(dev), r_hat.to(dev)
            restored = model(blurred, r_hat)
            # both images at the size the VLM sees
            restored = to_vlm_size(restored, cfg["image"]["long_side"])
            sharp = to_vlm_size(sharp, cfg["image"]["long_side"])
            if cfg["loss"] == "feature":
                part = feature_loss(encoder, reader.proc.image_processor, reader.size, dtype, restored, sharp)
            elif cfg["loss"] == "pixel":
                part = pixel_loss(restored, sharp)
            else:
                part = text_loss(restored, list(texts))
            (part / accumulate).backward()
            total += part.item() / accumulate
        loss = torch.tensor(total)
        # one optimisation step; clipping guards against large gradients near the zeros of H
        torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
        opt.step()
        model.project()
        # validate at fixed intervals and at the end; otherwise log the loss every 10 iterations
        if it % v["every"] == 0 or it == t["iterations"]:
            checkpoint(it, loss.item(), lr_factor)
        elif it % 10 == 0:
            log.writerow([it, round(loss.item(), 8), lr_factor, "", round(time.time() - t0)])
            log_file.flush()
            print(f"iteration {it}: loss {loss.item():.6f}, {time.time() - t0:.0f} s", flush=True)
    log_file.close()
