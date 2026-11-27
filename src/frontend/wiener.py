"""Learnable Wiener deconvolution with an estimated disc radius (method (d)).

For a blurred photograph y with estimated radius r_hat, the filter computes

    r      = s(r_hat) * r_hat                                  (radius correction)
    X_hat  = conj(H_r) Y / (|H_r|^2 + K(f; r))                  (Wiener filter)
    K      = exp(g(u, r)),   u = r * |f|                         (frequency-dependent regularisation)

where Y and H_r are the 2-D FFTs of y and of the disc PSF of radius r, and |f| is the frequency in
cycles per pixel. The learned parameters are two tables on fixed grids:

    log_k[b, m]   value of g at radius bin b and frequency knot u_m   (n_bins x n_knots)
    log_scale[b]  log of the radius correction s at radius bin b        (n_bins)

Between grid points both are interpolated linearly (in log r across bins, in u across knots); outside
the grid the nearest edge value is used. With log_k = log(k0) everywhere and log_scale = 0 the filter
is the ordinary Wiener filter with constant K = k0.

The same class is used for training (torch, on the GPU) and at test time (src/restore/learned_wiener.py),
so both compute exactly the same thing.
"""
import json
import math

import numpy as np
import torch
import torch.nn as nn


def radius_grid(r_min, r_max, count):
    """Centres of the radius bins, evenly spaced in log r.

    Args:
        r_min: smallest bin centre in pixels.
        r_max: largest bin centre in pixels.
        count: number of bins.

    Returns:
        List of floats.
    """
    return [float(r) for r in np.exp(np.linspace(math.log(r_min), math.log(r_max), count))]


def knot_grid(u_max, count):
    """Frequency knots u = r * |f|, evenly spaced from 0 to u_max.

    Args:
        u_max: largest knot; beyond it the last value is kept.
        count: number of knots.

    Returns:
        List of floats.
    """
    return [float(u) for u in np.linspace(0.0, u_max, count)]


def bin_weights(r, centres):
    """Linear interpolation weights of each radius over the radius bins, in log r.

    Args:
        r: tensor (B,) of radii in pixels.
        centres: tensor (n_bins,) of increasing bin centres.

    Returns:
        Tensor (B, n_bins): each row has at most two non-zero weights, summing to 1.
    """
    # work in log r and stay inside the outermost bins
    lr = torch.log(r.clamp(min=1e-3)).clamp(math.log(centres[0]), math.log(centres[-1]))
    lc = torch.log(centres)
    # index of the bin centre just above each radius (1 .. n_bins-1)
    hi = torch.searchsorted(lc, lr.contiguous()).clamp(1, len(centres) - 1)
    lo = hi - 1
    # position between the two centres, 0 at lo and 1 at hi
    t = (lr - lc[lo]) / (lc[hi] - lc[lo])
    w = torch.zeros(r.shape[0], len(centres), device=r.device, dtype=r.dtype)
    w.scatter_(1, lo[:, None], (1 - t)[:, None])
    w.scatter_add_(1, hi[:, None], t[:, None])
    return w


def interp_knots(u, knots, values):
    """Piecewise-linear interpolation of one curve per image, evaluated at many frequencies.

    Args:
        u: tensor (B, ...) of frequency coordinates u = r * |f|.
        knots: tensor (n_knots,) of increasing knot positions, starting at 0.
        values: tensor (B, n_knots): the curve of each image at the knots.

    Returns:
        Tensor shaped like u: the interpolated values; beyond the last knot the last value.
    """
    shape = u.shape
    u = u.reshape(shape[0], -1).clamp(0, knots[-1].item())
    # index of the knot just above each u (1 .. n_knots-1)
    hi = torch.searchsorted(knots, u.contiguous()).clamp(1, len(knots) - 1)
    lo = hi - 1
    t = (u - knots[lo]) / (knots[hi] - knots[lo])
    # gather the two neighbouring knot values of each image's own curve
    v_lo = torch.gather(values, 1, lo)
    v_hi = torch.gather(values, 1, hi)
    return (v_lo * (1 - t) + v_hi * t).reshape(shape)


def disc_psf(r, shape):
    """Uniform disc PSFs centred at pixel (0, 0) with wrap-around, ready for the FFT.

    The disc has value 1 inside radius r and 0 outside; a pixel at distance d from the centre takes
    the weight clamp(r + 0.5 - d, 0, 1), a linear approximation of the fraction of its area inside the
    circle (rule 3 of docs/tasks-david.md). The soft edge makes the PSF differentiable in r.

    Args:
        r: tensor (B,) of radii in pixels.
        shape: (height, width) of the padded image.

    Returns:
        Tensor (B, height, width), each PSF summing to 1.
    """
    h, w = shape
    # distance of every pixel from (0, 0), measured the short way round the periodic grid
    y = torch.arange(h, device=r.device, dtype=r.dtype)
    x = torch.arange(w, device=r.device, dtype=r.dtype)
    y = torch.minimum(y, h - y)
    x = torch.minimum(x, w - x)
    d = torch.sqrt(y[:, None] ** 2 + x[None, :] ** 2)
    # soft-edged disc, normalised to sum 1
    psf = (r[:, None, None] + 0.5 - d[None]).clamp(0, 1)
    return psf / psf.sum(dim=(1, 2), keepdim=True)


class LearnedWiener(nn.Module):
    """Wiener filter with a learned regularisation K(f; r) and a learned radius correction s(r)."""

    def __init__(self, radius_bins, u_knots, init_k=3e-3, pad=256, log_k_limits=(-13.8, 2.3),
                 log_scale_limit=0.2):
        """Create the filter at its starting point, the ordinary Wiener filter with K = init_k.

        Args:
            radius_bins: increasing list of radius-bin centres in pixels.
            u_knots: increasing list of frequency knots u = r * |f|, starting at 0.
            init_k: the constant K the filter starts from.
            pad: pixels of reflection padding on every side before the FFT; must exceed the
                largest radius so that the blur of the border does not wrap around.
            log_k_limits: (low, high) range that log K is kept in after each update
                (default: K between 1e-6 and 10).
            log_scale_limit: |log s| is kept below this after each update (0.2: radius within +-22 %).
        """
        super().__init__()
        # fixed grids, stored with the module so they move to the GPU with it
        self.register_buffer("radius_bins", torch.tensor(radius_bins, dtype=torch.float32))
        self.register_buffer("u_knots", torch.tensor(u_knots, dtype=torch.float32))
        # learned tables, starting from the ordinary Wiener filter
        self.log_k = nn.Parameter(torch.full((len(radius_bins), len(u_knots)), math.log(init_k)))
        self.log_scale = nn.Parameter(torch.zeros(len(radius_bins)))
        self.pad = pad
        self.log_k_limits = log_k_limits
        self.log_scale_limit = log_scale_limit

    def corrected_radius(self, r_hat):
        """Apply the learned radius correction r = s(r_hat) * r_hat.

        Args:
            r_hat: tensor (B,) of estimated radii in pixels.

        Returns:
            Tensor (B,) of corrected radii.
        """
        w = bin_weights(r_hat, self.radius_bins)
        return r_hat * torch.exp(w @ self.log_scale)

    def forward(self, y, r_hat):
        """Deconvolve a batch of blurred photographs.

        Args:
            y: tensor (B, H, W), float32, blurred photographs from load_tif() (paper about 1).
            r_hat: tensor (B,) of estimated disc radii in pixels.

        Returns:
            Tensor (B, H, W), float32: the restored images (not clipped).
        """
        b, h, w = y.shape
        # reflection padding keeps the border from wrapping around in the FFT
        yp = torch.nn.functional.pad(y[:, None], (self.pad,) * 4, mode="reflect")[:, 0]
        hp, wp = yp.shape[-2:]
        # corrected radius and the matching disc PSF
        r = self.corrected_radius(r_hat)
        H = torch.fft.rfft2(disc_psf(r, (hp, wp)))
        Y = torch.fft.rfft2(yp)
        # |f| in cycles per pixel on the half-spectrum grid of rfft2
        fy = torch.fft.fftfreq(hp, device=y.device)
        fx = torch.fft.rfftfreq(wp, device=y.device)
        f = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
        # K(f; r): each image's curve over u (mixing its two radius bins), evaluated at u = r |f|
        curves = bin_weights(r, self.radius_bins) @ self.log_k
        K = torch.exp(interp_knots(r[:, None, None] * f[None], self.u_knots, curves))
        # Wiener filter, then back to the image domain and remove the padding
        X = torch.conj(H) * Y / (H.real ** 2 + H.imag ** 2 + K)
        x = torch.fft.irfft2(X, s=(hp, wp))
        return x[:, self.pad:self.pad + h, self.pad:self.pad + w]

    @torch.no_grad()
    def project(self):
        """Keep the parameters inside their allowed ranges; called after every optimiser step."""
        self.log_k.clamp_(*self.log_k_limits)
        self.log_scale.clamp_(-self.log_scale_limit, self.log_scale_limit)

    def to_dict(self):
        """Everything needed to rebuild the filter, as plain lists.

        Returns:
            Dict that json.dump() can write.
        """
        return {"format": "learned_wiener v1",
                "radius_bins": self.radius_bins.tolist(),
                "u_knots": self.u_knots.tolist(),
                "log_k": self.log_k.detach().cpu().tolist(),
                "log_scale": self.log_scale.detach().cpu().tolist(),
                "pad": self.pad,
                "log_k_limits": list(self.log_k_limits),
                "log_scale_limit": self.log_scale_limit}

    def save(self, path):
        """Write the parameters as a JSON file.

        Args:
            path: output file.
        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def load(cls, path):
        """Rebuild a filter from a JSON file written by save().

        Args:
            path: parameter file.

        Returns:
            LearnedWiener with the stored grids and parameters.
        """
        with open(path) as f:
            p = json.load(f)
        m = cls(p["radius_bins"], p["u_knots"], pad=p["pad"], log_k_limits=tuple(p["log_k_limits"]),
                log_scale_limit=p["log_scale_limit"])
        with torch.no_grad():
            m.log_k.copy_(torch.tensor(p["log_k"]))
            m.log_scale.copy_(torch.tensor(p["log_scale"]))
        return m


def build_from_config(w):
    """Create the starting filter from a training config's `wiener` section.

    Args:
        w: dict with radius_bins {min, max, count}, u_knots {max, count}, init_k, pad,
            log_k_limits, log_scale_limit.

    Returns:
        LearnedWiener at its starting point.
    """
    rb, uk = w["radius_bins"], w["u_knots"]
    return LearnedWiener(radius_grid(rb["min"], rb["max"], rb["count"]), knot_grid(uk["max"], uk["count"]),
                         init_k=w["init_k"], pad=w["pad"], log_k_limits=tuple(w["log_k_limits"]),
                         log_scale_limit=w["log_scale_limit"])


@torch.no_grad()
def restore_numpy(model, a, r_hat):
    """Apply the filter to one NumPy image, as the evaluation pipeline needs.

    Args:
        model: LearnedWiener, on any device.
        a: 2-D float array from load_tif().
        r_hat: estimated radius in pixels (float).

    Returns:
        2-D float32 NumPy array of the same shape, on the CPU.
    """
    dev = model.radius_bins.device
    y = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(dev)[None]
    r = torch.tensor([r_hat], dtype=torch.float32, device=dev)
    return model(y, r)[0].cpu().numpy()
