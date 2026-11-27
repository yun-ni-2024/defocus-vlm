"""Compare Task 2 radius estimates on held-out HDC2021 training images."""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.data.hdc import train_samples  # noqa: E402
from src.data.image import load_tif  # noqa: E402
from src.psf import estimate_radius  # noqa: E402
from src.psf.estimate import edge_radius  # noqa: E402
from src.psf_backup.estimate import estimate_radius as estimate_spectrum  # noqa: E402

METHODS = {
    "combined": estimate_spectrum,
    "rings": lambda a: estimate_spectrum(a, method="rings"),
    "edges": edge_radius,
    "hybrid": estimate_radius,
}


def run(output, split, limit, exclude_first, check_repeat):
    radii = json.loads((REPO / "assets/radius_table.json").read_text())["radius"]
    samples = [s for s in train_samples() if (s.number % 10 == 0) == (split == "val")]
    groups = {}
    for sample in samples:
        groups.setdefault((sample.step, sample.font), []).append(sample)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with (output / "estimates.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("step", "font", "number", "method", "reference_px", "estimate_px", "seconds"))
        for (step, font), images in sorted(groups.items()):
            images.sort(key=lambda s: s.number)
            selected = images[exclude_first:][:limit]
            for index, sample in enumerate(selected):
                a = load_tif(sample.blurred)
                ref = float(radii[str(step)])
                for name, estimator in METHODS.items():
                    start = time.perf_counter()
                    estimate = estimator(a)
                    seconds = time.perf_counter() - start
                    if estimate is not None and (not math.isfinite(estimate) or estimate < 0):
                        raise ValueError(f"{name} returned invalid radius for {sample.blurred.name}")
                    writer.writerow((step, font, sample.number, name, ref, estimate, seconds))
                    rows.append((step, name, ref, estimate, seconds))
                if check_repeat and index == 0:
                    first = estimate_radius(a)
                    second = estimate_radius(a)
                    if first != second:
                        raise ValueError(f"Nondeterministic estimate for {sample.blurred.name}")
            f.flush()
            print(f"step {step:2d} {font}: {len(selected)} images", flush=True)

    summary = []
    for step in sorted({row[0] for row in rows}):
        for name in METHODS:
            group = [row for row in rows if row[:2] == (step, name)]
            differences = [estimate - ref for _, _, ref, estimate, _ in group if estimate is not None]
            relative = [abs(estimate - ref) / ref for _, _, ref, estimate, _ in group
                        if estimate is not None and ref > 0]
            summary.append({
                "step": step, "method": name, "n": len(group), "n_valid": len(differences),
                "median_signed_px": float(np.median(differences)) if differences else None,
                "median_absolute_px": float(np.median(np.abs(differences))) if differences else None,
                "median_absolute_relative": float(np.median(relative)) if relative else None,
                "median_seconds": float(np.median([row[-1] for row in group])),
                "p95_seconds": float(np.quantile([row[-1] for row in group], 0.95)),
            })
    data = {
        "split": split, "limit_per_step_font": limit,
        "excluded_first_per_step_font": exclude_first,
        "repeat_check_per_step_font": check_repeat, "results": summary,
    }
    (output / "summary.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("dev", "val"), default="val")
    parser.add_argument("--limit-per-step-font", type=int)
    parser.add_argument("--exclude-first-per-step-font", type=int, default=0)
    parser.add_argument("--check-repeat", action="store_true")
    args = parser.parse_args()
    if args.limit_per_step_font is not None and args.limit_per_step_font <= 0:
        parser.error("--limit-per-step-font must be positive")
    if args.exclude_first_per_step_font < 0:
        parser.error("--exclude-first-per-step-font cannot be negative")
    run(args.output, args.split, args.limit_per_step_font,
        args.exclude_first_per_step_font, args.check_repeat)


if __name__ == "__main__":
    main()
