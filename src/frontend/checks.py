"""Correctness checks for method (d), run before training (scripts/check_frontend.py).

1. split:      the validation pairs follow the rule of method (c): 3600 training, 400 validation.
2. wiener:     the filter at its starting point equals an independent NumPy Wiener filter with the
               same constant K and disc PSF.
3. restore:    the slot-1 module (src/restore/learned_wiener.py) gives the same image as the filter
               used in training.
4. vlm_input:  the differentiable input path matches the real to_pil() + processor: same grid, and
               visual tokens close to those of the processor.
5. gradients:  both losses give finite, non-zero gradients for both parameter tables.
"""
import json
import tempfile

import numpy as np
import torch

from src.data.image import load_tif, to_pil
from src.frontend.pairs import cache_key, split_pairs
from src.frontend.train import feature_loss, pixel_loss
from src.frontend.vlm_input import pixel_values, to_vlm_size, visual_tokens
from src.frontend.wiener import build_from_config, restore_numpy


def numpy_wiener(a, r, k, pad):
    """Ordinary Wiener filter with a constant K, written independently in NumPy.

    Args:
        a: 2-D float image.
        r: disc radius in pixels.
        k: the constant K.
        pad: reflection padding on every side.

    Returns:
        2-D float array of the same shape.
    """
    ap = np.pad(a.astype(np.float64), pad, mode="reflect")
    h, w = ap.shape
    # the disc PSF with the same soft edge, centred at (0, 0) with wrap-around
    y = np.minimum(np.arange(h), h - np.arange(h))
    x = np.minimum(np.arange(w), w - np.arange(w))
    d = np.sqrt(y[:, None] ** 2 + x[None, :] ** 2)
    psf = np.clip(r + 0.5 - d, 0, 1)
    psf /= psf.sum()
    H = np.fft.fft2(psf)
    X = np.conj(H) * np.fft.fft2(ap) / (np.abs(H) ** 2 + k)
    return np.real(np.fft.ifft2(X))[pad:pad + a.shape[0], pad:pad + a.shape[1]]


def check_split(cfg):
    """Check 1: the training/validation split.

    Args:
        cfg: the training config.

    Returns:
        (passed, message).
    """
    train, val = split_pairs(cfg["data"].get("steps"), cfg["data"]["val_every"])
    ok = len(train) == 3600 and len(val) == 400 and all(s.number % 10 == 0 for s in val)
    return ok, f"{len(train)} training pairs, {len(val)} validation pairs"


def check_wiener(cfg, sample, r_hat, device):
    """Check 2: the starting filter equals the independent NumPy Wiener filter.

    Args:
        cfg: the training config.
        sample: a training Sample.
        r_hat: its estimated radius.
        device: "cuda" or "cpu".

    Returns:
        (passed, message).
    """
    a = load_tif(sample.blurred, cfg["image"]["white_percentile"])
    model = build_from_config(cfg["wiener"]).to(device)
    ours = restore_numpy(model, a, r_hat)
    ref = numpy_wiener(a, r_hat, cfg["wiener"]["init_k"], cfg["wiener"]["pad"])
    err = float(np.abs(ours - ref).max())
    return err < 1e-3, f"largest difference {err:.2e} (image values about 0 to 1)"


def check_restore(cfg, sample, r_hat, device):
    """Check 3: the slot-1 module reproduces the training filter, through a saved parameter file.

    The filter's tables are perturbed first, so the check also covers saving and loading them.

    Args:
        cfg: the training config.
        sample: a training Sample.
        r_hat: its estimated radius.
        device: "cuda" or "cpu".

    Returns:
        (passed, message).
    """
    from src.restore.learned_wiener import build

    a = load_tif(sample.blurred, cfg["image"]["white_percentile"])
    model = build_from_config(cfg["wiener"]).to(device)
    # random but plausible tables, so the check is not only about the starting point
    with torch.no_grad():
        model.log_k.add_(0.5 * torch.randn_like(model.log_k))
        model.log_scale.add_(0.05 * torch.randn_like(model.log_scale))
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/params.json"
        model.save(path)
        # a cache holding this photograph's radius, so the module uses the same r_hat
        cache = f"{tmp}/radii.json"
        with open(cache, "w") as f:
            json.dump({"estimator": cfg["data"]["estimator"], "radius": {cache_key(sample.blurred): r_hat}}, f)
        restorer = build(path, estimator=cfg["data"]["estimator"], radius_cache=cache, device=device)
        slot = restorer(a, sample)
    err = float(np.abs(slot - restore_numpy(model, a, r_hat)).max())
    return err < 1e-5, f"largest difference {err:.2e}"


def check_vlm_input(cfg, samples, reader):
    """Check 4: the differentiable input path against the real to_pil() + processor.

    Args:
        cfg: the training config.
        samples: a few training Samples.
        reader: QwenReader.

    Returns:
        (passed, message).
    """
    ip, encoder, dtype = reader.proc.image_processor, reader.model.model.visual, reader.model.dtype
    lines, ok = [], True
    for s in samples:
        for cam in ("sharp", "blurred"):
            a = load_tif(getattr(s, cam), cfg["image"]["white_percentile"])
            # the real path
            real = ip(images=[to_pil(a, cfg["image"]["long_side"])], size=reader.size, return_tensors="pt")
            # the differentiable path
            x = to_vlm_size(torch.from_numpy(a)[None].to(reader.device), cfg["image"]["long_side"])
            pix, grid = pixel_values(x, ip, reader.size)
            same_grid = torch.equal(grid.cpu(), real["image_grid_thw"])
            ok &= same_grid
            # compare the visual tokens: error relative to the spread of the real tokens
            with torch.no_grad():
                t_real = visual_tokens(encoder, real["pixel_values"].to(reader.device),
                                       real["image_grid_thw"].to(reader.device), dtype).float()
                t_ours = visual_tokens(encoder, pix, grid, dtype).float()
            rel = float((t_ours - t_real).pow(2).mean() / t_real.var())
            pix_err = float((pix.cpu() - real["pixel_values"]).abs().mean())
            ok &= rel < 0.05
            lines.append(f"step {s.step} {cam}: grid {grid[0].tolist()} {'=' if same_grid else '!='} processor, "
                         f"mean pixel difference {pix_err:.4f}, token error {rel:.4f} of token variance")
    return ok, "\n    ".join(lines)


def check_gradients(cfg, sample, r_hat, reader):
    """Check 5: both losses give finite, non-zero gradients for both parameter tables.

    Args:
        cfg: the training config.
        sample: a training Sample.
        r_hat: its estimated radius.
        reader: QwenReader.

    Returns:
        (passed, message).
    """
    dev, wp = reader.device, cfg["image"]["white_percentile"]
    sharp = torch.from_numpy(load_tif(sample.sharp, wp))[None].to(dev)
    blurred = torch.from_numpy(load_tif(sample.blurred, wp))[None].to(dev)
    r = torch.tensor([r_hat], device=dev)
    lines, ok = [], True
    for name in ("feature", "pixel"):
        model = build_from_config(cfg["wiener"]).to(dev)
        restored = to_vlm_size(model(blurred, r), cfg["image"]["long_side"])
        target = to_vlm_size(sharp, cfg["image"]["long_side"])
        if name == "feature":
            loss = feature_loss(reader.model.model.visual, reader.proc.image_processor, reader.size,
                                reader.model.dtype, restored, target)
        else:
            loss = pixel_loss(restored, target)
        loss.backward()
        gk, gs = model.log_k.grad, model.log_scale.grad
        good = bool(torch.isfinite(gk).all() and torch.isfinite(gs).all() and gk.abs().sum() > 0 and gs.abs().sum() > 0)
        ok &= good
        lines.append(f"{name}: loss {loss.item():.4g}, |grad log_k| {gk.norm():.3g}, |grad log_scale| {gs.norm():.3g}")
    return ok, "\n    ".join(lines)
