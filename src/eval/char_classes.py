"""Which ground-truth characters a transcription got right, split into digits and letters.

The evaluation's character accuracy (metrics.char_accuracy) is one number per image and cannot say
which characters were missed. Here every ground-truth character is marked as read or not, so the
recall of digits and of letters can be compared (the training set has no digits, the test set does).

A character counts as read if it lies in a matching block of difflib.SequenceMatcher between the
ground truth and the prediction, with whitespace removed from both (as in the evaluation). This is
a different measure from the edit-distance accuracy, so the two are not directly comparable.
"""
import csv
import difflib

from src.eval.metrics import normalise

CLASSES = ("digits", "letters")


def read_mask(pred, truth):
    """Mark each ground-truth character as read or not.

    Args:
        pred: the model's transcription.
        truth: the correct transcription.

    Returns:
        (t, mask): the ground truth without whitespace, and a list of booleans of the same length,
        True where that character lies in a block matched with the prediction.
    """
    t, p = normalise(truth), normalise(pred)
    mask = [False] * len(t)
    # autojunk=False: with the default, frequent characters of long strings are ignored when matching
    for block in difflib.SequenceMatcher(None, t, p, autojunk=False).get_matching_blocks():
        for i in range(block.a, block.a + block.size):
            mask[i] = True
    return t, mask


def char_class(c):
    """Class of one ground-truth character.

    Args:
        c: a single non-whitespace character.

    Returns:
        "digits" for 0-9, otherwise "letters" (the HDC texts contain only letters and digits).
    """
    return "digits" if c.isdigit() else "letters"


def counts_per_step(predictions_csv):
    """Count read and total ground-truth characters per step and class.

    Args:
        predictions_csv: path to the predictions.csv of an evaluation run (columns step, truth,
            prediction, ...).

    Returns:
        Dict {(step, class): [read, total]} with integer steps.
    """
    counts = {}
    with open(predictions_csv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            t, mask = read_mask(row["prediction"], row["truth"])
            # add every ground-truth character to the counter of its step and class
            for c, ok in zip(t, mask):
                key = (int(row["step"]), char_class(c))
                n = counts.setdefault(key, [0, 0])
                n[0] += ok
                n[1] += 1
    return counts


def group_counts(counts, groups):
    """Add up per-step counts over groups of steps.

    Args:
        counts: dict from counts_per_step().
        groups: list of (first_step, last_step) pairs, both included.

    Returns:
        Dict {((first, last), class): [read, total]}.
    """
    out = {}
    for lo, hi in groups:
        for cls in CLASSES:
            # sum the counters of all steps inside the group
            read = sum(counts.get((s, cls), [0, 0])[0] for s in range(lo, hi + 1))
            total = sum(counts.get((s, cls), [0, 0])[1] for s in range(lo, hi + 1))
            out[((lo, hi), cls)] = [read, total]
    return out
