"""Training pairs for method (c): which sharp/blurred photographs to train on, and loading them."""
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.hdc import train_samples
from src.data.image import load_tif, to_pil


def split_pairs(source, steps=None, val_every=10):
    """Choose the training pairs and hold out a validation part.

    Args:
        source: "real" for the photograph pairs of the HDC2021 training set; "simulated" is
            added later for (c) simulated.
        steps: iterable of defocus steps to use, or None for all 20.
        val_every: samples whose number is a multiple of this are held out for validation
            (numbers run from 1 to 100 per step and font, so 10 gives 20 pairs per step).

    Returns:
        (train, val): two lists of Sample.
    """
    # only the real pairs exist so far
    if source != "real":
        raise NotImplementedError(f"pair source {source!r} is not implemented yet")
    samples = train_samples(steps=steps)
    # split by sample number, so every step and font is represented in both parts
    val = [s for s in samples if s.number % val_every == 0]
    train = [s for s in samples if s.number % val_every != 0]
    return train, val


class PairDataset(Dataset):
    """Loads the sharp and the blurred photograph of a sample as the images the VLM reads."""

    def __init__(self, samples, white_percentile=99.5, long_side=1024):
        """Keep the samples and the image settings.

        Args:
            samples: list of Sample.
            white_percentile: passed to load_tif().
            long_side: passed to to_pil().
        """
        self.samples = samples
        self.white_percentile = white_percentile
        self.long_side = long_side

    def __len__(self):
        """Return the number of pairs."""
        return len(self.samples)

    def __getitem__(self, i):
        """Load pair i.

        Args:
            i: index into the samples.

        Returns:
            (sharp, blurred): two PIL images, processed exactly as in the evaluation.
        """
        s = self.samples[i]
        # the same two steps as the evaluation pipeline with an empty slot 1
        sharp = to_pil(load_tif(s.sharp, self.white_percentile), self.long_side)
        blurred = to_pil(load_tif(s.blurred, self.white_percentile), self.long_side)
        return sharp, blurred


def _collate(batch):
    """Turn a list of (sharp, blurred) pairs into a list of sharp and a list of blurred images.

    Args:
        batch: list of (PIL image, PIL image).

    Returns:
        (list of sharp images, list of blurred images).
    """
    return [b[0] for b in batch], [b[1] for b in batch]


def endless_batches(samples, image_cfg, batch_size, seed=0, workers=0):
    """Yield batches of pairs forever, in a new random order every pass over the data.

    Args:
        samples: list of Sample to draw from.
        image_cfg: the config's image section (white_percentile, long_side).
        batch_size: pairs per batch.
        seed: seed of the shuffling, so a run can be repeated.
        workers: processes that load images in the background; 0 loads in the main process.

    Yields:
        (list of sharp PIL images, list of blurred PIL images), each of length batch_size.
    """
    ds = PairDataset(samples, image_cfg["white_percentile"], image_cfg["long_side"])
    # a seeded generator fixes the order of the shuffles
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, generator=g,
                        num_workers=workers, collate_fn=_collate, persistent_workers=workers > 0)
    # start a new pass whenever one is used up
    while True:
        for batch in loader:
            yield batch
