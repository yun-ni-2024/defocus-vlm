"""Evaluate the backup radius estimator against the backup radius table.

Development uses the held-out training images (sample number a multiple of 10: 20 per step); the
test set is used once, at the end. Several settings can be compared in one call: the spectra are
computed once per image and spectrum setting and reused.

Usage (environment activated; a CPU node is enough):
  python scripts/backup_eval_estimator.py <output folder> [split: val | test] ['<JSON list of settings>']
Writes estimates.csv (one row per image and setting) and prints a per-step summary per setting.
"""
import csv
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.data.hdc import test_samples, train_samples  # noqa: E402
from src.data.image import load_tif  # noqa: E402
from src.psf_backup.estimate import (DEFAULTS, build_reference, reference_file, radial_spectrum,  # noqa: E402
                                     search, signal_cutoff)

REPO = pathlib.Path(__file__).resolve().parents[1]
SPECTRUM_KEYS = ("window", "tukey_alpha", "axis_gap")


def estimate_from_spectrum(spec, p, ref_logp=None):
    """The estimator's search, given an already computed spectrum (same as estimate_radius()).

    Args:
        spec: (rho, logp, count) from radial_spectrum().
        p: full settings dict.
        ref_logp: reference log spectrum, needed unless p["method"] == "rings".

    Returns:
        Estimated radius (float).
    """
    return search(spec, p, ref_logp)


def reference_spectrum(key, source="sharp"):
    """Load a reference text spectrum, building it first from training photographs if needed.

    Only the training part is used (sample numbers that are not multiples of 10), never the
    validation images.
      source "sharp":    10 in-focus (CAM01) photographs per step, 200 in all;
      source "blurred0": the defocused camera (CAM02) at step 0, where the blur is about 1.6 px,
                         so the spectrum includes that camera's own lens and sensor; 180 photographs.

    Args:
        key: spectrum settings (window, tukey_alpha, axis_gap).
        source: "sharp" or "blurred0".

    Returns:
        Reference log spectrum per bin.
    """
    path = reference_file(source)
    if path.exists():
        z = np.load(path)
        if tuple(z["key"].tolist()) == tuple(str(k) for k in key):
            return z["logp"]
    if source == "sharp":
        per_step = {}
        for s in train_samples():
            if s.number % 10 != 0 and len(per_step.setdefault(s.step, [])) < 10:
                per_step[s.step].append(s)
        paths = [s.sharp for v in per_step.values() for s in v]
    else:
        paths = [s.blurred for s in train_samples(steps=[0]) if s.number % 10 != 0]
    print(f"building the {source} reference spectrum from {len(paths)} photographs", flush=True)
    rho, logp = build_reference((load_tif(x) for x in paths), *key)
    np.savez(path, rho=rho, logp=logp, key=np.array([str(k) for k in key]))
    return logp


def summarise(rows, table):
    """Print per-step accuracy of one setting and return the overall score.

    Args:
        rows: list of dicts with step and estimate.
        table: step -> reference radius.

    Returns:
        Mean over steps 2-19 of the median absolute relative error.
    """
    meds = []
    print(" step   ref    median est   med |rel err|   max |rel err|   within 5%")
    for s in range(20):
        est = np.array([r["estimate"] for r in rows if r["step"] == s])
        rel = np.abs(est - table[s]) / table[s]
        if s >= 2:
            meds.append(np.median(rel))
        print(f"  {s:3d} {table[s]:7.2f} {np.median(est):10.2f} {100 * np.median(rel):12.1f}% "
              f"{100 * rel.max():13.1f}% {100 * np.mean(rel < 0.05):10.0f}%")
    score = float(np.mean(meds))
    print(f"  mean over steps 2-19 of the median |relative error|: {100 * score:.2f}%")
    return score


def main():
    """Run every setting on every selected image and write the results."""
    out = pathlib.Path(sys.argv[1])
    split = sys.argv[2] if len(sys.argv) > 2 else "val"
    settings = json.loads(sys.argv[3]) if len(sys.argv) > 3 else [{}]
    out.mkdir(parents=True, exist_ok=True)
    table = {int(k): v for k, v in
             json.loads((REPO / "assets" / "radius_table_backup.json").read_text())["radius"].items()}
    samples = ([s for s in train_samples() if s.number % 10 == 0] if split == "val" else test_samples())
    print(f"{len(samples)} images ({split})", flush=True)

    # reference spectra for the settings that need one
    refs = {}
    for overrides in settings:
        p = {**DEFAULTS, **overrides}
        key = tuple(p[k] for k in SPECTRUM_KEYS)
        if p["method"] in ("reference", "combined") and (key, p["reference"]) not in refs:
            refs[(key, p["reference"])] = reference_spectrum(key, p["reference"])

    # spectra, computed once per image and spectrum setting
    spectra = {}
    rows = []
    t0 = time.time()
    for i, s in enumerate(samples):
        a = load_tif(s.blurred)
        for j, overrides in enumerate(settings):
            p = {**DEFAULTS, **overrides}
            key = tuple(p[k] for k in SPECTRUM_KEYS)
            if key not in spectra:
                spectra[key] = radial_spectrum(a, *key)
            est = estimate_from_spectrum(spectra[key], p, refs.get((key, p["reference"])))
            rows.append({"setting": j, "step": s.step, "font": s.font, "number": s.number,
                         "reference": table[s.step], "estimate": round(est, 3),
                         "cutoff": round(signal_cutoff(spectra[key][0], spectra[key][1], p["snr"], p["rho_min_bins"]), 4)})
        spectra.clear()
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(samples)} images, {time.time() - t0:.0f} s", flush=True)

    with open(out / "estimates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for j, overrides in enumerate(settings):
        print(f"\n=== setting {j}: {overrides}")
        summarise([r for r in rows if r["setting"] == j], table)


if __name__ == "__main__":
    main()
