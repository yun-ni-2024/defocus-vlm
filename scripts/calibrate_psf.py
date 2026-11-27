"""Estimate and cross-validate the disc radius of all 20 HDC2021 training steps."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.psf.calibration import check_dot, fit_line, read_tiff, sharp_line_kernel  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TARGETS = {"LSF_X": 1, "LSF_Y": 0}
CAM02_ONLY = "cam02_thin_line"
PAIRED = "paired_finite_width"
MIN_LINE_FIT = 0.5
MIN_PAIRED_GAIN = 0.002


def digest(path):
    return hashlib.sha256(path.read_bytes()).digest()


def summarize(fits):
    usable = [
        fit["radius"] for fit in fits
        if fit["explained_fraction"] >= MIN_LINE_FIT and fit["radius"] < 280
    ]
    return float(np.median(usable or [fit["radius"] for fit in fits]))


def measure_step(root, step):
    targets, dots, seen_lines, seen_dots = [], [], {}, set()
    for font in ("Times", "Verdana"):
        folder = root / "train" / f"step{step}" / font
        indices = {}
        for target, axis in TARGETS.items():
            name = f"focusStep_{step}_{target}.tif"
            blurred = folder / "CAM02" / name
            key = (target, digest(blurred))
            if key not in seen_lines:
                seen_lines[key] = len(targets)
                sharp = sharp_line_kernel(read_tiff(folder / "CAM01" / name), axis)
                photo = read_tiff(blurred)
                targets.append({
                    "font": font, "target": target, "axis": axis, "path": blurred, "sharp": sharp,
                    CAM02_ONLY: fit_line(photo, axis),
                    PAIRED: fit_line(photo, axis, sharp=sharp),
                })
            indices[target] = seen_lines[key]
        dot = folder / "CAM02" / f"focusStep_{step}_PSF.tif"
        key = digest(dot)
        if key not in seen_dots:
            seen_dots.add(key)
            dots.append((font, dot, indices))

    # Each candidate is estimated with the evaluated CAM02 target excluded.
    variants = (CAM02_ONLY, PAIRED)
    folds = []
    for index, held in enumerate(targets):
        train = targets[:index] + targets[index + 1:]
        photo = read_tiff(held["path"])
        fold = {"font": held["font"], "target": held["target"]}
        for variant in variants:
            radius = summarize([t[variant] for t in train])
            fit = fit_line(photo, held["axis"], sharp=held["sharp"], fixed_radius=radius)
            fold[variant] = {"radius": radius, "explained_fraction": fit["explained_fraction"]}
        folds.append(fold)

    scores = {
        name: float(np.mean([fold[name]["explained_fraction"] for fold in folds]))
        for name in variants
    }
    gain = scores[PAIRED] - scores[CAM02_ONLY]
    selected = PAIRED if gain >= MIN_PAIRED_GAIN else CAM02_ONLY
    full_radii = {name: summarize([t[name] for t in targets]) for name in variants}
    radius = full_radii[selected]
    line_radii = [t[selected]["radius"] for t in targets]
    spread = float(np.ptp(line_radii))
    flags = []
    if spread > max(4.0, 0.15 * radius):
        flags.append("line_estimates_disagree")
    if any(t[selected]["explained_fraction"] < MIN_LINE_FIT or t[selected]["radius"] >= 280 for t in targets):
        flags.append("weak_line_fit")
    if radius < 10 and selected == CAM02_ONLY:
        flags.append("finite_line_width_not_corrected")

    dot_checks = []
    for font, path, indices in dots:
        x = targets[indices["LSF_Y"]][selected]["centre"]
        y = targets[indices["LSF_X"]][selected]["centre"]
        dot_checks.append(dict(font=font, **check_dot(read_tiff(path), x, y, radius)))
    dot_scores = np.median([d["explained_fraction"] for d in dot_checks], axis=0)
    if max(dot_scores) < 0.1:
        flags.append("weak_dot_signal")
    elif dot_scores.argmax() != 1:
        flags.append("dot_prefers_an_alternative")
    confidence = "high"
    if flags or scores[selected] < 0.9:
        confidence = "medium"
    if radius < 10 or scores[selected] < 0.6 or "weak_line_fit" in flags:
        confidence = "low"

    return {
        "step": step, "radius": radius, "selected": selected, "candidates": full_radii,
        "held_out_score": scores, "paired_gain": gain, "line_spread": spread,
        "confidence": confidence, "flags": flags, "folds": folds, "dots": dot_checks,
        "lines": [{k: t[k] for k in ("font", "target", *variants)} for t in targets],
    }


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="HDC2021 directory containing train/")
    parser.add_argument("--output", type=Path, default=REPO / "assets/radius_table.json")
    parser.add_argument("--diagnostics", type=Path, default=REPO / "assets/radius_validation.json")
    args = parser.parse_args()
    if args.output.resolve() == args.diagnostics.resolve():
        parser.error("--output and --diagnostics must be different files")
    rows = []
    for step in range(20):
        row = measure_step(args.data_root, step)
        rows.append(row)
        print(f"step {step:2d}: {row['radius']:7.3f} px  {row['selected']}  {row['confidence']}", flush=True)
    table = {
        "units": "pixels of the 2360 x 1460 image",
        "source": "training set, blurred camera (CAM02) calibration targets, with CAM01 paired line validation",
        "radius": {str(row["step"]): round(row["radius"], 3) for row in rows},
    }
    write_json(args.diagnostics, {
        "validation": "leave one distinct CAM02 line image out per fold; the held-out CAM01 supplies only the focused template",
        "score": "mean fraction of background-adjusted line variation explained on held-out targets",
        "minimum_paired_gain": MIN_PAIRED_GAIN,
        "minimum_line_fit": MIN_LINE_FIT,
        "confidence": "heuristic quality labels; not statistical confidence intervals",
        "steps": rows,
    })
    write_json(args.output, table)


if __name__ == "__main__":
    main()
