"""Training pairs for method (d): full-resolution sharp/blurred photographs with their estimated radius.

The split is the same rule as method (c) (src/adapt/pairs.py on the c-real branch): samples whose
number is a multiple of val_every are held out, so (c) and (d) validate on the same 400 pairs.
"""
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.hdc import data_root, train_samples
from src.data.image import load_tif
from src.utils.run import defocus_root


def split_pairs(steps=None, val_every=10):
    """Choose the training pairs and hold out a validation part.

    Args:
        steps: iterable of defocus steps to use, or None for all 20.
        val_every: samples whose number is a multiple of this are held out for validation
            (numbers run from 1 to 100 per step and font, so 10 gives 20 pairs per step).

    Returns:
        (train, val): two lists of Sample.
    """
    samples = train_samples(steps=steps)
    # split by sample number, so every step and font is represented in both parts
    val = [s for s in samples if s.number % val_every == 0]
    train = [s for s in samples if s.number % val_every != 0]
    return train, val


def cache_key(path):
    """Key of a photograph in a radius cache: its path relative to the dataset folder.

    Args:
        path: path of a photograph under data_root().

    Returns:
        String with forward slashes, e.g. "train/step8/Times/CAM02/focusStep_8_timesR_size_30_sample_0001.tif".
    """
    return path.resolve().relative_to(data_root().resolve()).as_posix()


def load_radius_cache(path):
    """Read a radius cache written by scripts/estimate_radii.py.

    Args:
        path: cache file, relative to DEFOCUS_ROOT or absolute.

    Returns:
        (estimator name, dict from cache_key() to estimated radius).
    """
    with open(defocus_root() / path) as f:
        c = json.load(f)
    return c["estimator"], c["radius"]


class WienerPairDataset(Dataset):
    """Loads a pair at full resolution, with the estimated radius of its blurred photograph."""

    def __init__(self, samples, radii, white_percentile=99.5):
        """Keep the samples, their radii and the image setting.

        Args:
            samples: list of Sample.
            radii: dict from cache_key() of the blurred photograph to its estimated radius.
            white_percentile: passed to load_tif().
        """
        self.samples = samples
        self.radii = radii
        self.white_percentile = white_percentile

    def __len__(self):
        """Return the number of pairs."""
        return len(self.samples)

    def __getitem__(self, i):
        """Load pair i.

        Args:
            i: index into the samples.

        Returns:
            (sharp, blurred, radius, text): two float32 arrays of shape (1460, 2360) from load_tif(),
            the estimated radius of the blurred one and the correct transcription.
        """
        s = self.samples[i]
        sharp = load_tif(s.sharp, self.white_percentile)
        blurred = load_tif(s.blurred, self.white_percentile)
        return sharp, blurred, np.float32(self.radii[cache_key(s.blurred)]), s.text


def endless_batches(samples, radii, white_percentile, batch_size, seed=0, workers=0):
    """Yield batches of pairs forever, in a new random order every pass over the data.

    Args:
        samples: list of Sample to draw from.
        radii: dict from cache_key() to estimated radius.
        white_percentile: passed to load_tif().
        batch_size: pairs per batch.
        seed: seed of the shuffling, so a run can be repeated.
        workers: processes that load images in the background; 0 loads in the main process.

    Yields:
        (sharp, blurred, radius, texts): float32 tensors (B, 1460, 2360), (B, 1460, 2360), (B,) and
        a list of B transcriptions.
    """
    ds = WienerPairDataset(samples, radii, white_percentile)
    # a seeded generator fixes the order of the shuffles
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, generator=g,
                        num_workers=workers, persistent_workers=workers > 0)
    # start a new pass whenever one is used up
    while True:
        for batch in loader:
            yield batch
