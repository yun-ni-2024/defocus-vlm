"""Local backup of the physics deliverables.

calibrate.py measures the disc radius of every step from the calibration targets (backup of task 1);
estimate.py estimates the radius of a single photograph (backup of task 2), with the interface of
task 2:

    from src.psf_backup import estimate_radius
    r = estimate_radius(a)
"""
from src.psf_backup.estimate import estimate_radius  # noqa: F401
