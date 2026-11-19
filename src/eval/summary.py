"""Per-step accuracy with bootstrap confidence intervals."""
import csv
from collections import defaultdict

import numpy as np


def bootstrap_ci(values, n_boot=2000, level=0.95, seed=0):
    """Confidence interval of a mean by resampling the values with replacement.

    Args:
        values: per-image scores.
        n_boot: number of resamples.
        level: coverage of the interval.
        seed: random seed, fixed so the interval is reproducible.

    Returns:
        (low, high) as floats.
    """
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    # mean of each resample, then the matching percentiles
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def summarise(rows, score="accuracy"):
    """Average a score per defocus step.

    Args:
        rows: dicts with at least "step" and the score column (e.g. read from predictions.csv).
        score: name of the score column.

    Returns:
        List of dicts with step, n, mean, ci_low, ci_high, sorted by step.
    """
    by_step = defaultdict(list)
    for r in rows:
        by_step[int(r["step"])].append(float(r[score]))
    out = []
    for step in sorted(by_step):
        v = by_step[step]
        lo, hi = bootstrap_ci(v)
        out.append({"step": step, "n": len(v), "mean": float(np.mean(v)), "ci_low": lo, "ci_high": hi})
    return out


def write_summary(rows, path):
    """Write the output of summarise() as a CSV file.

    Args:
        rows: list of dicts from summarise().
        path: output file.

    Returns:
        None.
    """
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "n", "mean", "ci_low", "ci_high"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
