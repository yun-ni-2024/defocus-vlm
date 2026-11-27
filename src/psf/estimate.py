"""Blind radius estimates from single HDC2021 text photographs."""
from functools import lru_cache

import numpy as np

EDGE_MIN_COUNT = 6
EDGE_MAX_RADIUS = 18.0
SCAN_HALF_WIDTH = 24


@lru_cache(maxsize=1)
def _edge_templates():
    """Return centred profiles for an ideal step blurred by a uniform disc."""
    positions = np.arange(-SCAN_HALF_WIDTH, SCAN_HALF_WIDTH + 1)
    radii = np.arange(0.5, EDGE_MAX_RADIUS + 0.01, 0.25)
    u = np.clip(positions[None, :] / radii[:, None], -1, 1)
    templates = (np.arcsin(u) + u * np.sqrt(1 - u**2)) / np.pi
    templates -= templates.mean(axis=1, keepdims=True)
    return positions, radii, templates, np.sum(templates**2, axis=1)


def edge_radius(a):
    """Return a light-blur estimate, or None when too few isolated edges agree."""
    positions, radii, templates, norm = _edge_templates()
    candidates = []
    # Scan across vertical and horizontal edges, trying both polarities.
    for transposed, photo in enumerate((a, a.T)):
        height, width = photo.shape
        margin = max(SCAN_HALF_WIDTH + 5, width // 20)
        for y in range(height // 6, height - height // 6, 4):
            profile = photo[y - 2:y + 3].mean(axis=0)
            gradient = profile[8:] - profile[:-8]
            interior = gradient[margin:-margin]
            indices = np.argpartition(np.abs(interior), -8)[-8:]
            for i in indices:
                centre = int(i) + margin + 4
                observed = profile[centre + positions]
                if gradient[centre - 4] > 0:
                    observed = observed[::-1]
                left, right = observed[:5], observed[-5:]
                contrast = float(left.mean() - right.mean())
                if contrast < 0.12 or max(np.ptp(left), np.ptp(right)) > 0.08:
                    continue
                darkness = observed.mean() - observed
                amplitude = (templates @ darkness) / norm
                error = np.mean((amplitude[:, None] * templates - darkness)**2, axis=1) / contrast**2
                best = int(error.argmin())
                if error[best] < 0.008 and 0 < best < len(radii) - 1:
                    x0, y0 = (y, centre) if transposed else (centre, y)
                    candidates.append((float(error[best]), x0, y0, float(radii[best])))

    selected = []
    for _, x, y, radius in sorted(candidates):
        if all((x - old_x)**2 + (y - old_y)**2 > 40**2 for old_x, old_y, _ in selected):
            selected.append((x, y, radius))
        if len(selected) == 12:
            break
    if len(selected) < EDGE_MIN_COUNT:
        return None
    estimates = np.array([r for _, _, r in selected])
    centre = float(np.median(estimates))
    if np.median(np.abs(estimates - centre)) > max(2.0, 0.25 * centre):
        return None
    return centre


def estimate_radius(a):
    """Estimate the disc radius, with a local-edge check for light blur."""
    from src.psf_backup import estimate_radius as estimate_spectrum

    spectral = float(estimate_spectrum(a))
    # Edge fits are capped at 18 px. Only consult them when the spectrum
    # indicates light blur, and retain the spectral estimate near focus.
    if spectral <= 12.0:
        local = edge_radius(a)
        if local is not None and local >= 3.5:
            return local
    return spectral
