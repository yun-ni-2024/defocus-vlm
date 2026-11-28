"""DPIR-style HQS deconvolution using DeepInverse's pretrained grayscale DRUNet."""
import json
import math

import numpy as np
import torch
import torch.nn.functional as F

from src.psf import estimate_radius
from src.restore.wiener import disc_kernel
from src.utils.run import REPO


def denoise_tiles(image, denoiser, sigma, tile_size, halo):
    """Run the denoiser with context around each tile and keep its centre."""
    height, width = image.shape[-2:]
    result = torch.empty_like(image)
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            bottom, right = min(y + tile_size, height), min(x + tile_size, width)
            top_ctx, left_ctx = max(0, y - halo), max(0, x - halo)
            patch = image[..., top_ctx:min(height, bottom + halo), left_ctx:min(width, right + halo)]
            clean = denoiser(patch.contiguous(), sigma)
            result[..., y:bottom, x:right] = clean[
                ..., y - top_ctx:bottom - top_ctx, x - left_ctx:right - left_ctx
            ]
    return result


@torch.inference_mode()
def deconvolve(a, radius, denoiser, device, schedule, tile_size, halo):
    """Alternate an FFT data update and denoising at the scheduled noise level."""
    kernel = disc_kernel(radius)
    if kernel.shape == (1, 1):
        return np.asarray(a, dtype=np.float32)

    height, width = a.shape
    pad = kernel.shape[0] // 2
    photo = torch.as_tensor(a, dtype=torch.float32, device=device)[None, None]
    observed = F.pad(photo, (pad, pad, pad, pad), mode="reflect")
    psf = torch.zeros_like(observed[0, 0])
    kh, kw = kernel.shape
    psf[:kh, :kw] = torch.as_tensor(kernel, dtype=psf.dtype, device=device)
    transfer = torch.fft.rfft2(torch.roll(psf, (-pad, -pad), dims=(0, 1)))
    power = transfer.abs().square()
    data = transfer.conj() * torch.fft.rfft2(observed)
    restored = observed

    for sigma, penalty in schedule:
        numerator = data + penalty * torch.fft.rfft2(restored)
        update = torch.fft.irfft2(numerator / (power + penalty), s=observed.shape[-2:])
        restored = denoise_tiles(update, denoiser, sigma, tile_size, halo)

    return restored[0, 0, pad:pad + height, pad:pad + width].cpu().numpy()


def build(
    radius="estimated",
    measurement_sigma=0.01,
    sigma_start=49 / 255,
    sigma_end=0.01,
    iterations=8,
    tile_size=512,
    halo=64,
    device="cuda",
):
    """Build the evaluator restorer using a geometric DRUNet noise schedule."""
    if radius not in ("estimated", "calibrated"):
        raise ValueError("radius must be 'estimated' or 'calibrated'")
    noise = (measurement_sigma, sigma_start, sigma_end)
    if not all(math.isfinite(value) and 0 < value <= 1 for value in noise):
        raise ValueError("noise levels must be finite and in (0, 1]")
    if sigma_end > sigma_start:
        raise ValueError("sigma_end cannot exceed sigma_start")
    if type(iterations) is not int or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if type(tile_size) is not int or tile_size < 32 or type(halo) is not int or halo < 0:
        raise ValueError("tile_size must be at least 32 and halo must be nonnegative")

    from deepinv.models import DRUNet

    denoiser = DRUNet(
        in_channels=1, out_channels=1, pretrained="download", device=device
    ).eval()
    sigmas = np.geomspace(sigma_start, sigma_end, iterations)
    schedule = [(float(sigma), float(0.23 * (measurement_sigma / sigma)**2)) for sigma in sigmas]
    table = None
    if radius == "calibrated":
        table = json.loads((REPO / "assets/radius_table.json").read_text())["radius"]

    def restore(a, sample):
        r = estimate_radius(a) if table is None else float(table[str(sample.step)])
        return deconvolve(a, r, denoiser, device, schedule, tile_size, halo)

    return restore
