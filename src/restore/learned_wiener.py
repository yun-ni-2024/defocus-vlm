"""Slot 1 for method (d): the trained Wiener filter with the radius estimated from the photograph.

Config:
    restore:
      name: learned_wiener
      params: assets/d_feature/params.json      # written by training, relative to the repository root
      estimator: psf_backup                      # radius estimator: src/<estimator>.estimate_radius
      radius_cache: cache/radii_test_psf_backup.json   # optional, relative to DEFOCUS_ROOT
      device: cuda

The filter itself is src/frontend/wiener.py, the same code that was trained.
"""
import importlib

import torch

from src.frontend.pairs import cache_key, load_radius_cache
from src.frontend.wiener import LearnedWiener, restore_numpy
from src.utils.run import REPO


def build(params, radius="estimated", estimator="psf_backup", radius_cache=None, device="cuda"):
    """Create the restorer from a trained parameter file.

    Args:
        params: parameter file written by training (params/it<N>.json), relative to the repository
            root or absolute.
        radius: only "estimated" is supported: the radius comes from the photograph alone.
        estimator: package under src/ that provides estimate_radius(a), e.g. "psf_backup".
        radius_cache: optional cache from scripts/estimate_radii.py, made with the same estimator;
            photographs found in it are not estimated again (the estimator is deterministic).
        device: where the filter runs, "cuda" or "cpu".

    Returns:
        A callable restorer(a, sample) -> array.
    """
    if radius != "estimated":
        raise ValueError("learned_wiener uses the estimated radius only")
    # the trained filter
    model = LearnedWiener.load(REPO / params).to(device if torch.cuda.is_available() else "cpu").eval()
    estimate_radius = importlib.import_module(f"src.{estimator}").estimate_radius
    # radii already computed for these photographs, if a cache is given
    cached = {}
    if radius_cache is not None:
        made_by, cached = load_radius_cache(radius_cache)
        if made_by != estimator:
            raise ValueError(f"radius cache made by {made_by}, config asks for {estimator}")

    def restore(a, sample):
        """Estimate the radius of the photograph and deconvolve it with the trained filter.

        Args:
            a: float image from load_tif().
            sample: its Sample; only used to look the photograph up in the radius cache.

        Returns:
            2-D float32 array of the same shape (to_pil() clips it to [0, 1]).
        """
        # the radius from the photograph alone (cached or computed now)
        key = cache_key(sample.blurred)
        r_hat = cached[key] if key in cached else estimate_radius(a)
        return restore_numpy(model, a, r_hat)

    return restore
