"""Scores for comparing a transcription with the ground truth."""
import re

from rapidfuzz.distance import Levenshtein


def normalise(s, keep_whitespace=False):
    """Prepare a string for comparison.

    Args:
        s: transcription or ground truth.
        keep_whitespace: if False, all spaces and line breaks are removed; if True, any run of
            whitespace becomes one space and the ends are stripped.

    Returns:
        The normalised string.
    """
    if keep_whitespace:
        return re.sub(r"\s+", " ", s).strip()
    return re.sub(r"\s+", "", s)


def char_accuracy(pred, truth, keep_whitespace=False):
    """Character accuracy: 1 - edit distance / length of the ground truth, clipped at 0.

    Whether whitespace counts is still open (docs/evaluation.md), so it is a parameter.

    Args:
        pred: the model's transcription.
        truth: the correct transcription.
        keep_whitespace: passed to normalise().

    Returns:
        Float in [0, 1]; 1 means an exact match.
    """
    p, t = normalise(pred, keep_whitespace), normalise(truth, keep_whitespace)
    # Levenshtein distance: insertions, deletions and substitutions all cost 1
    return max(0.0, 1.0 - Levenshtein.distance(p, t) / max(1, len(t)))
