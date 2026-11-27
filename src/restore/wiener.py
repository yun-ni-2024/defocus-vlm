"""Full-resolution Wiener deconvolution with a uniform disc PSF."""
import json
import math

import numpy as np

from src.psf import estimate_radius
from src.utils.run import REPO


def disc_kernel(radius):
    """Return a centred, unit-sum disc with fractional boundary pixels."""
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("disc radius must be finite and nonnegative")
    if radius <= 0.5:
        return np.ones((1, 1))

    extent = math.ceil(radius)
    x = np.arange(-extent, extent + 1, dtype=float)
    r2 = radius**2

    def primitive(u):
        return (u * np.sqrt(np.maximum(0, r2 - u**2))
                + r2 * np.arcsin(np.clip(u / radius, 0, 1))) / 2

    def area(x_edge, y_edge):
        # Signed area from the origin to a pixel corner, inside the disc.
        x_abs = np.minimum(np.abs(x_edge), radius)
        y_abs = np.minimum(np.abs(y_edge), radius)
        flat = np.minimum(x_abs, np.sqrt(np.maximum(0, r2 - y_abs**2)))
        return np.sign(x_edge) * np.sign(y_edge) * (
            flat * y_abs + primitive(x_abs) - primitive(flat)
        )

    hi, lo = x + 0.5, x - 0.5
    kernel = (area(hi[None, :], hi[:, None]) - area(lo[None, :], hi[:, None])
              - area(hi[None, :], lo[:, None]) + area(lo[None, :], lo[:, None]))
    np.maximum(kernel, 0, out=kernel)
    return kernel / kernel.sum()


def deconvolve(a, radius, regularization):
    """Apply the constant-regularization Wiener filter to one photograph."""
    kernel = disc_kernel(radius)
    if kernel.shape == (1, 1):
        return np.asarray(a, dtype=np.float32)

    pad = kernel.shape[0] // 2
    padded = np.pad(a, pad, mode="reflect")
    psf = np.zeros(padded.shape, dtype=np.float32)
    psf[:kernel.shape[0], :kernel.shape[1]] = kernel
    psf = np.roll(psf, (-pad, -pad), axis=(0, 1))
    transfer = np.fft.rfft2(psf)
    power = transfer.real**2 + transfer.imag**2
    gain = transfer.conj() / (power + regularization)
    gain[0, 0] = 1  # Preserve the paper's average brightness.
    restored = np.fft.irfft2(np.fft.rfft2(padded) * gain, s=padded.shape)
    return restored[pad:pad + a.shape[0], pad:pad + a.shape[1]].astype(np.float32)


def build(radius="estimated", regularization=0.003):
    """Build the evaluation restorer using an estimated or calibrated radius."""
    if radius not in ("estimated", "calibrated"):
        raise ValueError("radius must be 'estimated' or 'calibrated'")
    if not math.isfinite(regularization) or regularization <= 0:
        raise ValueError("regularization must be finite and positive")
    table = None
    if radius == "calibrated":
        table = json.loads((REPO / "assets/radius_table.json").read_text())["radius"]

    def restore(a, sample):
        r = estimate_radius(a) if table is None else float(table[str(sample.step)])
        return deconvolve(a, r, regularization)

    return restore
