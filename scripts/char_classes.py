"""Recall of digits and of letters for evaluation results (src/eval/char_classes.py).

Reads predictions.csv of each given result folder and prints a CSV table to standard output:
one row per method, group of steps and character class. Runs on a laptop, no GPU or model needed.

Usage:
  python scripts/char_classes.py assets/results/baseline_a assets/results/c_real assets/results/d_text
  python scripts/char_classes.py --groups 0-4,5-14,15-19 assets/results/*
Without --groups every step is its own group.
"""
import argparse
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.eval.char_classes import CLASSES, counts_per_step, group_counts  # noqa: E402


def parse_groups(text):
    """Parse a list of step ranges.

    Args:
        text: e.g. "0-4,5-14,15-19"; a single number is a group of one step.

    Returns:
        List of (first_step, last_step) pairs.
    """
    groups = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        groups.append((int(lo), int(hi or lo)))
    return groups


def main():
    """Count, group and print the recall table."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="result folders, each with a predictions.csv")
    ap.add_argument("--groups", default=None, help="step ranges such as 0-4,5-14,15-19 (default: each step)")
    args = ap.parse_args()

    out = csv.writer(sys.stdout, lineterminator="\n")
    out.writerow(["method", "steps", "class", "read", "total", "recall"])
    for folder in map(pathlib.Path, args.results):
        counts = counts_per_step(folder / "predictions.csv")
        # default: one group per step that occurs in the results
        groups = parse_groups(args.groups) if args.groups else [(s, s) for s in sorted({s for s, _ in counts})]
        grouped = group_counts(counts, groups)
        for lo, hi in groups:
            for cls in CLASSES:
                read, total = grouped[((lo, hi), cls)]
                steps = str(lo) if lo == hi else f"{lo}-{hi}"
                recall = f"{read / total:.3f}" if total else ""
                out.writerow([folder.name, steps, cls, read, total, recall])


if __name__ == "__main__":
    main()
