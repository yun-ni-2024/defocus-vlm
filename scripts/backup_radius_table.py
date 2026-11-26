"""Backup of task 1: measure the disc radius of every defocus step from the line targets.

For each step, the LSF_X and LSF_Y targets of the training set's blurred camera (CAM02) are fitted
with the in-focus camera's (CAM01) line convolved with a uniform disc (src/psf_backup/calibrate.py).
Each step has a copy of the targets in the Times and in the Verdana folder; copies that are the
same file are used once. The radius of a step is the mean of its measurements.

Usage (environment activated; a CPU node is enough):
  python scripts/backup_radius_table.py assets/radius_table_backup.json
Writes the table in the format of assets/radius_table.json, with every single measurement and the
settings added, and prints one line per step.
"""
import hashlib
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.data.hdc import data_root  # noqa: E402
from src.psf_backup.calibrate import cross_section, fit_radius, line_kernel  # noqa: E402

# fit settings, chosen by the robustness checks of 2026-11-25
SETTINGS = {"search": 120, "window": 380, "degree": 4, "margin": 380}
TARGETS = {"LSF_X": 1, "LSF_Y": 0}           # target name -> array axis along its line


def file_hash(path):
    """MD5 of a file, to recognise copies of the same photograph.

    Args:
        path: file to hash.

    Returns:
        Hex digest string.
    """
    return hashlib.md5(path.read_bytes()).hexdigest()


def measure_step(step, root):
    """Fit every distinct line target of one step.

    Args:
        step: defocus step, 0 to 19.
        root: HDC2021 folder.

    Returns:
        List of dicts, one per distinct photograph pair: font, target, r, y0, ink, rms.
    """
    out, seen = [], set()
    for font in ["Times", "Verdana"]:
        for target, axis in TARGETS.items():
            d = root / "train" / f"step{step}" / font
            sharp_path = d / "CAM01" / f"focusStep_{step}_{target}.tif"
            blurred_path = d / "CAM02" / f"focusStep_{step}_{target}.tif"
            # the same file in both font folders counts once
            key = (target, file_hash(blurred_path))
            if key in seen:
                continue
            seen.add(key)
            sharp = cross_section(sharp_path, axis, SETTINGS["margin"])
            blurred = cross_section(blurred_path, axis, SETTINGS["margin"])
            k, c = line_kernel(sharp)
            res = fit_radius(blurred, k, c, search=SETTINGS["search"], window=SETTINGS["window"],
                             degree=SETTINGS["degree"])
            out.append({"font": font, "target": target, "r": round(res["r"], 2),
                        "y0": round(res["y0"], 1), "ink": round(res["ink"], 3),
                        "rms": round(res["rms"], 2)})
    return out


def main():
    """Measure all 20 steps and write the table."""
    out_path = pathlib.Path(sys.argv[1])
    root = data_root()
    radius, measurements = {}, {}
    for step in range(20):
        ms = measure_step(step, root)
        rs = np.array([m["r"] for m in ms])
        radius[str(step)] = round(float(rs.mean()), 2)
        measurements[str(step)] = ms
        print(f"step {step:2d}: r = {rs.mean():7.2f}  (n = {len(rs)}, min {rs.min():.2f}, max {rs.max():.2f})  "
              + "  ".join(f"{m['font'][0]}{m['target'][-1]} {m['r']:.1f}" for m in ms), flush=True)
    table = {"units": "pixels of the 2360 x 1460 image",
             "source": "training set, blurred camera (CAM02), calibration targets LSF_X and LSF_Y "
                       "(local backup measurement)",
             "radius": radius,
             "method": "line target cross-section fitted with the in-focus line convolved with a "
                       "uniform disc plus a polynomial background; mean over targets and distinct copies. "
                       "The horizontal cross-section (LSF_Y) is consistently about 3 px wider than the "
                       "vertical one (LSF_X), also on the PSF dot, so the blur is slightly anisotropic; "
                       "the radius is the best single disc, their mean. Removing the line tilt before "
                       "averaging changed r by less than 0.1 px (0.3 px at steps 0-1), so it is not done.",
             "settings": SETTINGS,
             "measurements": measurements}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(table, indent=2) + "\n")


if __name__ == "__main__":
    main()
