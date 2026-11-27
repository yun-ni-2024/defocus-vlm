"""Estimate the disc radius of a defocused photograph from its power spectrum (blind, no training).

A photograph is the sharp scene convolved with the PSF, plus noise; in the frequency domain

    Y(f) = X(f) H_r(f) + N(f),   H_r(rho) = 2 J1(2 pi r rho) / (2 pi r rho)

for a uniform disc of radius r (rho in cycles per pixel). H_r crosses zero on rings at
rho = 3.83 / (2 pi r), 7.02 / (2 pi r), ..., whatever the text is, so the photograph's power spectrum
has dark rings whose positions depend only on r. The radially averaged log power spectrum is

    L(rho) = log |X|^2 + log |H_r|^2   (where the signal is above the noise)

and log |X|^2 of text varies smoothly with rho. Two ways of using this are combined:

  rings:     for each candidate r, L - log(|H_r|^2 + eta) is fitted with a smooth curve (a polynomial
             in log rho); the r whose rings leave the smallest residual wins. Very precise when the
             rings are above the noise (radius above about 45 px), but for smaller radii only the
             main lobe is visible and it underestimates, and it sometimes takes the second ring for
             the first.
  reference: the unknown |X|^2 is replaced by the average spectrum of in-focus training photographs
             (all text has the same size), and P = c S |H_r|^2 + N is fitted. This also uses the
             main lobe, so it works for medium radii, but it collapses for the widest blur.

choose() takes the ring estimate when the two agree (or the reference one collapsed on a strongly
blurred photo) and the reference estimate otherwise. On the held-out training images (2026-11-25)
the median relative error is below 4 % for steps 5-19 and 7.5 % at step 4; for steps 2-3 the
estimate is about half the calibrated radius: there the real PSF is not disc-like enough for the
spectrum to show the calibrated disc.
"""
from pathlib import Path

import numpy as np
import torch

DEFAULTS = {
    "method": "combined",   # "rings", "reference" or "combined" (see choose())
    "reference": "sharp",   # reference spectrum: "sharp" (in-focus camera) or "blurred0" (defocused camera, step 0)
    "agree_tol": 0.10,      # combined: the two estimates agree if they differ by less than this fraction
    "collapse_r": 3.0,      # combined: a reference estimate below this is treated as failed ...
    "sharp_cutoff": 0.05,   # ... unless the signal reaches this frequency (cycles/px): then the photo is nearly sharp
    "window": "tukey",     # "hann" or "tukey": tapers the image edges so they add no cross to the spectrum
    "tukey_alpha": 0.25,    # fraction of each side that is tapered (Tukey window)
    "axis_gap": 3,          # frequency bins on each side of the two axes left out (text line regularity)
    "eta": 1e-3,            # floor added to |H_r|^2: the rings are never perfectly dark
    "degree": 6,            # degree of the smooth polynomial in log rho
    "snr": 2.0,             # use frequencies whose power exceeds the noise floor by this factor (natural log)
    "rho_min_bins": 3,      # lowest frequency used, in bins of 1 / (long image side)
    "r_min": 1.0,           # smallest radius searched
    "r_max": 260.0,         # largest radius searched
}


def _window_1d(n, kind, alpha):
    """1-D taper of length n.

    Args:
        n: length.
        kind: "hann" or "tukey".
        alpha: tapered fraction for the Tukey window.

    Returns:
        Float array of length n, 1 in the middle and falling to 0 at the ends.
    """
    if kind == "hann":
        return np.hanning(n)
    # Tukey: flat in the middle, cosine taper over alpha / 2 of the length at each end
    x = np.linspace(0, 1, n)
    w = np.ones(n)
    edge = alpha / 2
    lo, hi = x < edge, x > 1 - edge
    w[lo] = 0.5 * (1 - np.cos(np.pi * x[lo] / edge))
    w[hi] = 0.5 * (1 - np.cos(np.pi * (1 - x[hi]) / edge))
    return w


def radial_spectrum(a, window="tukey", tukey_alpha=0.25, axis_gap=3):
    """Radially averaged log power spectrum of an image.

    Args:
        a: 2-D float image.
        window: taper applied before the FFT ("hann" or "tukey").
        tukey_alpha: tapered fraction for the Tukey window.
        axis_gap: bins on each side of the frequency axes that are left out.

    Returns:
        (rho, logp, count): bin centres in cycles per pixel (bin width 1 / long side), the mean
        log power in each bin, and the number of frequencies averaged in each bin.
    """
    h, w = a.shape
    # remove the mean and taper the edges
    win = np.outer(_window_1d(h, window, tukey_alpha), _window_1d(w, window, tukey_alpha))
    F = np.fft.rfft2((a - a.mean()) * win)
    logp = np.log(np.abs(F) ** 2 + 1e-12)
    # radial frequency of every coefficient
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    rho = np.sqrt(fx ** 2 + fy ** 2)
    # leave out the axes (window cross, regular text lines) and frequencies beyond Nyquist
    keep = (np.abs(fx) * w > axis_gap) & (np.abs(fy) * h > axis_gap) & (rho <= 0.5)
    width = 1.0 / max(h, w)
    idx = np.floor(rho[keep] / width).astype(int)
    count = np.bincount(idx)
    total = np.bincount(idx, weights=logp[keep])
    ok = count > 0
    centres = (np.arange(len(count)) + 0.5) * width
    return centres[ok], total[ok] / count[ok], count[ok]


def disc_otf(r, rho):
    """Frequency response of a uniform disc of radius r.

    Args:
        r: radius in pixels.
        rho: array of radial frequencies in cycles per pixel.

    Returns:
        Array 2 J1(x) / x with x = 2 pi r rho (1 at rho = 0).
    """
    x = 2 * np.pi * r * np.asarray(rho, dtype=np.float64)
    j1 = torch.special.bessel_j1(torch.from_numpy(x)).numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(x > 1e-8, 2 * j1 / x, 1.0)
    return out


def score_radii(rho, logp, count, radii, eta=1e-3, degree=6, snr=2.0, rho_min_bins=3):
    """Residual of the ring model for every candidate radius.

    Args:
        rho, logp, count: from radial_spectrum().
        radii: candidate radii.
        eta: floor added to |H_r|^2.
        degree: degree of the smooth polynomial in log rho.
        snr: frequencies are used up to the last one whose log power exceeds the noise floor by snr.
        rho_min_bins: number of lowest bins left out.

    Returns:
        (rss, used): weighted residual sum of squares per radius, and the boolean mask of bins used.
    """
    # fit only up to where the signal sinks into the noise
    hi = signal_cutoff(rho, logp, snr, rho_min_bins)
    used = (np.arange(len(rho)) >= rho_min_bins) & (rho <= hi)
    x, y, wgt = np.log(rho[used]), logp[used], count[used].astype(np.float64)
    # smooth part: polynomial in log rho, scaled to [-1, 1]
    t = 2 * (x - x.min()) / max(x.max() - x.min(), 1e-9) - 1
    B = np.vander(t, degree + 1, increasing=True)
    sw = np.sqrt(wgt)[:, None]
    Bw = B * sw
    Q, _ = np.linalg.qr(Bw)
    rss = np.empty(len(radii))
    for i, r in enumerate(radii):
        # what is left after removing the rings of radius r must be smooth
        resid = (y - np.log(disc_otf(r, rho[used]) ** 2 + eta)) * sw[:, 0]
        resid = resid - Q @ (Q.T @ resid)
        rss[i] = resid @ resid
    return rss, used


def reference_file(source="sharp"):
    """Path of the stored reference spectrum.

    Args:
        source: "sharp" or "blurred0" (see DEFAULTS).

    Returns:
        Path under assets/.
    """
    return Path(__file__).resolve().parents[2] / "assets" / f"text_spectrum_backup_{source}.npz"


def choose(r_rings, r_ref, cutoff, agree_tol=0.10, collapse_r=3.0, sharp_cutoff=0.05):
    """Combine the two estimates.

    The ring method is the more precise one when its rings are the right ones, but for small radii
    the rings sink into the noise and it underestimates, and sometimes it takes the second ring for
    the first (giving 0.55 times the radius). The reference method handles small and medium radii,
    but collapses to about 1 px for the widest blur.

    Args:
        r_rings: estimate of the ring method.
        r_ref: estimate of the reference method.
        cutoff: highest frequency where the signal is above the noise (signal_cutoff()).
        agree_tol: relative difference below which the two agree.
        collapse_r: reference estimates below this count as failed ...
        sharp_cutoff: ... unless the signal reaches this frequency: a nearly sharp photograph, for
            which a small radius is right and the ring method finds no rings.

    Returns:
        The combined estimate: the ring estimate when the two agree or the reference one failed,
        otherwise the reference estimate.
    """
    if r_ref < collapse_r:
        return r_ref if cutoff >= sharp_cutoff else r_rings
    if abs(r_rings - r_ref) <= agree_tol * r_ref:
        return r_rings
    return r_ref


def signal_cutoff(rho, logp, snr=2.0, rho_min_bins=3):
    """Highest frequency at which the photograph's spectrum is clearly above the noise floor.

    Args:
        rho, logp: from radial_spectrum().
        snr: required excess over the noise floor, in natural-log units of power.
        rho_min_bins: frequencies below this bin are not considered.

    Returns:
        Frequency in cycles per pixel.
    """
    # noise floor: the highest frequencies of a blurred photograph hold only noise
    floor = np.median(logp[rho > 0.4])
    # smooth the spectrum a little so single noisy bins do not count
    smooth = np.convolve(logp, np.ones(9) / 9, mode="same")
    above = np.nonzero(smooth > floor + snr)[0]
    return float(rho[above[-1]]) if len(above) else float(rho[rho_min_bins + 10])


def build_reference(images, window="tukey", tukey_alpha=0.25, axis_gap=3):
    """Average radial log power spectrum of in-focus text photographs.

    All photographs have the same size and the same text size, so their spectra are alike; the
    average stands for the unknown sharp spectrum |X|^2 of a new photograph.

    Args:
        images: iterable of in-focus photographs (load_tif() arrays, all of the same shape).
        window, tukey_alpha, axis_gap: as in radial_spectrum().

    Returns:
        (rho, logp): bin centres and mean log power per bin.
    """
    total, n = None, 0
    for a in images:
        rho, logp, _ = radial_spectrum(a, window, tukey_alpha, axis_gap)
        total = logp if total is None else total + logp
        n += 1
    return rho, total / n


def score_reference(rho, logp, count, ref_logp, radii, rho_min_bins=3):
    """Residual of the reference-spectrum model for every candidate radius.

    Model in linear power: P(rho) = c * S(rho) * |H_r(rho)|^2 + N, with S the reference spectrum,
    c a contrast factor and N the noise floor. For each r, c and N are solved by least squares with
    relative weights (equivalent to fitting the log spectrum), then the residual of log P is taken.

    Args:
        rho, logp, count: from radial_spectrum() of the blurred photograph.
        ref_logp: reference log spectrum on the same bins (build_reference()).
        radii: candidate radii.
        rho_min_bins: number of lowest bins left out.

    Returns:
        Weighted residual sum of squares of log P per radius.
    """
    used = np.arange(len(rho)) >= rho_min_bins
    rho_u, y, wgt = rho[used], logp[used], count[used].astype(np.float64)
    P, S = np.exp(y), np.exp(ref_logp[used])
    rss = np.empty(len(radii))
    for i, r in enumerate(radii):
        A = S * disc_otf(r, rho_u) ** 2
        # weighted least squares for (c, N) with weights 1 / P^2, i.e. relative errors
        M = np.column_stack([A, np.ones_like(A)]) / P[:, None] * np.sqrt(wgt)[:, None]
        coef, *_ = np.linalg.lstsq(M, np.sqrt(wgt), rcond=None)
        c, noise = max(coef[0], 1e-30), max(coef[1], 1e-30)
        resid = y - np.log(c * A + noise)
        rss[i] = np.sum(wgt * resid ** 2)
    return rss


def estimate_radius(a, **params):
    """Estimate the disc radius of a blurred photograph.

    Args:
        a: blurred photograph as returned by load_tif() (NumPy float array, paper about 1).
        **params: optional overrides of DEFAULTS, for tuning.

    Returns:
        Estimated radius in pixels of the full-resolution image (float), as defined in rule 3.
    """
    p = {**DEFAULTS, **params}
    spec = radial_spectrum(a, p["window"], p["tukey_alpha"], p["axis_gap"])
    ref = np.load(reference_file(p["reference"]))["logp"] if p["method"] != "rings" else None
    return search(spec, p, ref)


def search(spec, p, ref=None):
    """Find the best radius for a spectrum with the chosen method.

    Args:
        spec: (rho, logp, count) from radial_spectrum().
        p: full settings dict (DEFAULTS with overrides).
        ref: reference log spectrum, needed unless p["method"] == "rings".

    Returns:
        Estimated radius (float).
    """
    rho, logp, count = spec

    def best(method):
        """Coarse search on a geometric grid (1 % steps), then a fine one around the best radius."""
        def score(radii):
            if method == "reference":
                return score_reference(rho, logp, count, ref, radii, p["rho_min_bins"])
            return score_radii(rho, logp, count, radii, p["eta"], p["degree"], p["snr"], p["rho_min_bins"])[0]
        coarse = np.exp(np.arange(np.log(p["r_min"]), np.log(p["r_max"]), 0.01))
        r0 = coarse[int(np.argmin(score(coarse)))]
        fine = np.linspace(r0 * 0.98, r0 * 1.02, 41)
        return float(fine[int(np.argmin(score(fine)))])

    if p["method"] == "combined":
        cutoff = signal_cutoff(rho, logp, p["snr"], p["rho_min_bins"])
        return choose(best("rings"), best("reference"), cutoff, p["agree_tol"], p["collapse_r"], p["sharp_cutoff"])
    return best(p["method"])
