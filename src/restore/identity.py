"""A restorer that returns the image unchanged.

Used to check the slot itself: a run with `restore: {name: identity}` must give exactly the same
predictions as a run with `restore: null`.
"""


def build():
    """Create the identity restorer.

    Returns:
        A callable restorer(a, sample) -> array that returns `a` unchanged.
    """
    def restore(a, sample):
        """Return the image unchanged.

        Args:
            a: float image from load_tif().
            sample: its Sample (unused).

        Returns:
            The same array.
        """
        return a

    return restore
