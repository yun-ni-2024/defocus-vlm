"""Slot 1 of the evaluation pipeline: image processing between load_tif() and to_pil().

Each method lives in its own module src/restore/<name>.py, which defines

    build(**params) -> restorer
    restorer(a, sample) -> array

where `a` is the float image from load_tif() (paper about 1) and `sample` is its Sample. A config
selects a method with

    restore:
      name: <name>
      <param>: <value>

and `restore: null` (the default) leaves the slot empty.

Test-time methods must use only the photograph itself. `sample` is passed so that reference
methods, such as deconvolution with the calibrated PSF of the right step, can read `sample.step`.
"""
import importlib


def build_restorer(cfg):
    """Create the restorer named in a config's `restore` section.

    Args:
        cfg: the `restore` section of the config: None, or a dict with "name" and the method's
            parameters.

    Returns:
        A callable restorer(a, sample) -> array, or None when the slot is empty.
    """
    # empty slot: the image goes straight to the VLM
    if cfg is None:
        return None
    # everything except the name is passed to the method's build()
    params = {k: v for k, v in cfg.items() if k != "name"}
    module = importlib.import_module(f"src.restore.{cfg['name']}")
    return module.build(**params)
