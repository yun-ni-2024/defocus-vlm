"""Full-resolution deconvolution with isotropic total variation and ADMM."""
import json
import math

import numpy as np
import torch
import torch.nn.functional as F

from src.psf import estimate_radius
from src.restore.wiener import disc_kernel
from src.utils.run import REPO


def gradient(image):
    """Periodic forward differences along rows and columns."""
    return torch.stack((
        torch.roll(image, -1, dims=0) - image,
        torch.roll(image, -1, dims=1) - image,
    ))


def gradient_adjoint(field):
    """Adjoint of the periodic forward differences."""
    dy, dx = field
    return torch.roll(dy, 1, dims=0) - dy + torch.roll(dx, 1, dims=1) - dx


@torch.inference_mode()
def deconvolve(a, radius, tv_weight, rho, iterations, device):
    """Minimize half the squared data error plus isotropic total variation."""
    kernel = disc_kernel(radius)
    if kernel.shape == (1, 1):
        return np.asarray(a, dtype=np.float32)

    h, w = a.shape
    pad = kernel.shape[0] // 2
    photo = torch.as_tensor(a, device=device, dtype=torch.float32)
    y = F.pad(photo[None, None], (pad, pad, pad, pad), mode="reflect")[0, 0]
    height, width = y.shape
    psf = torch.zeros_like(y)
    psf[:kernel.shape[0], :kernel.shape[1]] = torch.as_tensor(
        kernel, device=device, dtype=y.dtype
    )
    transfer = torch.fft.rfft2(torch.roll(psf, (-pad, -pad), dims=(0, 1)))
    ky = torch.fft.fftfreq(height, device=y.device, dtype=y.dtype)
    kx = torch.fft.rfftfreq(width, device=y.device, dtype=y.dtype)
    laplacian = 4 * (torch.sin(torch.pi * ky[:, None]).square()
                     + torch.sin(torch.pi * kx[None, :]).square())
    denominator = transfer.abs().square() + rho * laplacian
    data_term = transfer.conj() * torch.fft.rfft2(y)

    z = gradient(y)
    u = torch.zeros_like(z)
    for _ in range(iterations):
        rhs = data_term + rho * torch.fft.rfft2(gradient_adjoint(z - u))
        x = torch.fft.irfft2(rhs / denominator, s=y.shape)
        dx = gradient(x)
        p = dx + u
        magnitude = torch.linalg.vector_norm(p, dim=0).clamp_min(1e-12)
        z = p * (1 - tv_weight / (rho * magnitude)).clamp_min(0)
        u = p - z

    return x[pad:pad + h, pad:pad + w].cpu().numpy()


def build(radius="estimated", tv_weight=0.001, rho=0.05, iterations=30, device="cuda"):
    """Build the evaluator's ADMM restorer with a blind or calibrated radius."""
    if radius not in ("estimated", "calibrated"):
        raise ValueError("radius must be 'estimated' or 'calibrated'")
    if not all(math.isfinite(value) and value > 0 for value in (tv_weight, rho)):
        raise ValueError("tv_weight and rho must be finite and positive")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    table = None
    if radius == "calibrated":
        table = json.loads((REPO / "assets/radius_table.json").read_text())["radius"]

    def restore(a, sample):
        r = estimate_radius(a) if table is None else float(table[str(sample.step)])
        return deconvolve(a, r, tv_weight, rho, iterations, device)

    return restore
