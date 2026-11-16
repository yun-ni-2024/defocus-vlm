"""Index the HDC2021 text images: which sharp and blurred files belong together and what they say.

Only text images are indexed; calibration targets, natural photos and the QR code are skipped.
Folder layout (see docs/dataset.md):
  train/step<N>/<Times|Verdana>/CAM01|CAM02/focusStep_<N>_<font>_size_30_sample_<k>.tif
  test/steps_<a>_to_<b>/CAM01_focused|CAM02_blurred/focusStep_<N>_<font>_size_30_sample_<k>.tif
"""
import getpass
import os
import re
from dataclasses import dataclass
from pathlib import Path

# text image file names; the font tag is "timesR" or "verdanaRef" in both splits
TEXT_FILE = re.compile(r"^focusStep_(\d+)_(timesR|verdanaRef)_size_30_sample_(\d+)\.tif$")
FONTS = {"timesR": "Times", "verdanaRef": "Verdana"}


@dataclass(frozen=True)
class Sample:
    """One text scene photographed by both cameras.

    Attributes:
        split: "train" or "test".
        step: defocus step of the blurred camera, 0 to 19.
        font: "Times" or "Verdana".
        number: sample number from the file name.
        sharp: path of the in-focus photograph (CAM01).
        blurred: path of the defocused photograph (CAM02).
        text: the correct transcription, lines separated by "\\n".
    """
    split: str
    step: int
    font: str
    number: int
    sharp: Path
    blurred: Path
    text: str


def data_root():
    """Return the HDC2021 folder, found from $DEFOCUS_ROOT.

    Returns:
        Path of <DEFOCUS_ROOT>/data/hdc2021; DEFOCUS_ROOT defaults to /w/20261/<username>/csc2529.
    """
    base = os.environ.get("DEFOCUS_ROOT", os.path.join("/w/20261", getpass.getuser(), "csc2529"))
    return Path(base) / "data" / "hdc2021"


def _pairs(blur_dir, sharp_dir, split, steps):
    """Pair every blurred text image in a folder with its sharp version and transcription.

    Args:
        blur_dir: folder of CAM02 images.
        sharp_dir: folder of the matching CAM01 images.
        split: "train" or "test", stored in each sample.
        steps: set of steps to keep, or None for all.

    Returns:
        List of Sample, sorted by file name.
    """
    out = []
    for blurred in sorted(blur_dir.glob("*.tif")):
        # skip everything that is not a text image
        m = TEXT_FILE.match(blurred.name)
        if m is None:
            continue
        step, font, number = int(m.group(1)), FONTS[m.group(2)], int(m.group(3))
        if steps is not None and step not in steps:
            continue
        # the sharp image and the transcription share the file name
        sharp = sharp_dir / blurred.name
        text = sharp.with_suffix(".txt").read_text().strip()
        out.append(Sample(split, step, font, number, sharp, blurred, text))
    return out


def test_samples(steps=None, root=None):
    """List the text images of the test set.

    Args:
        steps: iterable of defocus steps to keep, or None for all 20.
        root: HDC2021 folder; defaults to data_root().

    Returns:
        List of Sample, sorted by step, then font, then number (40 per step).
    """
    root = Path(root) if root else data_root()
    keep = set(steps) if steps is not None else None
    out = []
    # five steps share one folder; the step is read from the file name
    for folder in sorted((root / "test").glob("steps_*")):
        out += _pairs(folder / "CAM02_blurred", folder / "CAM01_focused", "test", keep)
    return sorted(out, key=lambda s: (s.step, s.font, s.number))


def train_samples(steps=None, root=None):
    """List the text images of the training set.

    Args:
        steps: iterable of defocus steps to keep, or None for all 20.
        root: HDC2021 folder; defaults to data_root().

    Returns:
        List of Sample, sorted by step, then font, then number (200 per step).
    """
    root = Path(root) if root else data_root()
    keep = set(steps) if steps is not None else None
    out = []
    # one folder per step and font
    for font_dir in sorted((root / "train").glob("step*/*")):
        out += _pairs(font_dir / "CAM02", font_dir / "CAM01", "train", keep)
    return sorted(out, key=lambda s: (s.step, s.font, s.number))
