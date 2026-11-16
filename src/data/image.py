"""Read HDC2021 photographs and turn them into images a VLM can take."""
import numpy as np
import tifffile
from PIL import Image


def load_tif(path, white_percentile=99.5):
    """Read a 16-bit greyscale TIFF and scale it so the white paper is about 1.

    Args:
        path: TIFF file.
        white_percentile: the pixel value at this percentile is mapped to 1 (the paper background).

    Returns:
        2-D float32 array, mostly in [0, 1]; values above 1 are kept.
    """
    # 16-bit integers to floats
    a = tifffile.imread(path).astype(np.float32)
    # divide by a bright percentile instead of the maximum, so a few hot pixels do not set the scale
    return a / np.percentile(a, white_percentile)


def to_pil(a, long_side=1024):
    """Convert a [0, 1] greyscale array to an 8-bit RGB PIL image, optionally downscaled.

    Args:
        a: 2-D float array; values outside [0, 1] are clipped.
        long_side: the longer image side is reduced to this many pixels; None keeps the full size.

    Returns:
        PIL.Image in RGB mode.
    """
    # clip, quantise to 8 bits and copy the grey channel into three
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).convert("RGB")
    # shrink only, never enlarge
    if long_side is not None and max(im.size) > long_side:
        s = long_side / max(im.size)
        im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.LANCZOS)
    return im
