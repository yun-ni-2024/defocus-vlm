"""Pretrained DeepInverse RAM with a uniform-disc blur operator."""
import json
import math

import numpy as np
import torch
import torch.nn.functional as F

from src.psf import estimate_radius
from src.restore.wiener import disc_kernel
from src.utils.run import REPO


@torch.inference_mode()
def deconvolve(a, radius, model, blur_fft, sigma, tile_size, context, device):
    """Reconstruct full-resolution tiles and keep each tile’s centre."""
    kernel = disc_kernel(radius)
    if kernel.shape == (1, 1):
        return np.asarray(a, dtype=np.float32)

    height, width = a.shape
    halo = kernel.shape[0] // 2 + context
    photo = torch.as_tensor(a, dtype=torch.float32, device=device)[None, None]
    padded = F.pad(photo, (halo, halo, halo, halo), mode="reflect")
    psf = torch.as_tensor(kernel, dtype=photo.dtype, device=device)[None, None]
    result = torch.empty_like(photo)
    for y in range(0, height, tile_size):
        h = min(tile_size, height - y)
        for x in range(0, width, tile_size):
            w = min(tile_size, width - x)
            patch = padded[..., y:y + h + 2 * halo, x:x + w + 2 * halo]
            # RAM uses four spatial scales. Pad before constructing the blur
            # operator so it sees the same dimensions as the model.
            extra_h = -patch.shape[-2] % 16
            extra_w = -patch.shape[-1] % 16
            patch = F.pad(patch, (0, extra_w, 0, extra_h), mode="reflect")
            physics = blur_fft(patch.shape[1:], filter=psf, device=device)
            restored = model(patch, physics=physics, sigma=sigma)
            result[..., y:y + h, x:x + w] = restored[
                ..., halo:halo + h, halo:halo + w
            ]
    return result[0, 0].cpu().numpy()


def build(radius="estimated", sigma=0.01, tile_size=512, context=64, device="cuda"):
    """Create a RAM restorer with blind or calibrated blur radii."""
    if radius not in ("estimated", "calibrated"):
        raise ValueError("radius must be 'estimated' or 'calibrated'")
    if not math.isfinite(sigma) or not 0 <= sigma <= 1:
        raise ValueError("sigma must be finite and in [0, 1]")
    if type(tile_size) is not int or tile_size < 32:
        raise ValueError("tile_size must be an integer of at least 32")
    if type(context) is not int or context < 16:
        raise ValueError("context must be an integer of at least 16")

    from deepinv.models import RAM
    from deepinv.physics import BlurFFT

    model = RAM(pretrained=True, device=device).eval()
    table = None
    if radius == "calibrated":
        table = json.loads((REPO / "assets/radius_table.json").read_text())["radius"]

    def restore(a, sample):
        r = estimate_radius(a) if table is None else float(table[str(sample.step)])
        return deconvolve(a, r, model, BlurFFT, sigma, tile_size, context, device)

    return restore
