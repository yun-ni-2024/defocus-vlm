"""Fit a uniform-disc blur to CAM02 line calibration targets."""
import numpy as np
from PIL import Image


def read_tiff(path):
    with Image.open(path) as image:
        return np.asarray(image, dtype=np.float64)


def disc_profile(distance, radius):
    """Integrate the projected disc over each pixel of a line cross-section."""
    if radius == 0:
        return (np.abs(distance) < 0.5).astype(float)
    left = np.clip((distance - 0.5) / radius, -1, 1)
    right = np.clip((distance + 0.5) / radius, -1, 1)
    return (
        np.arcsin(right) + right * np.sqrt(1 - right**2)
        - np.arcsin(left) - left * np.sqrt(1 - left**2)
    ) / np.pi


def line_section(photo, axis):
    """Average along the line while excluding the outer image margins."""
    oriented = photo if axis == 1 else photo.T
    height, width = oriented.shape
    return oriented[:, 380:width - 380].mean(axis=1), height // 2 + (30 if axis == 1 else 15)


def sharp_line_kernel(photo, axis):
    """Recover the finite-width absorption profile from the focused target."""
    section, guess = line_section(photo, axis)
    centre = guess - 80 + int(section[guess - 80:guess + 81].argmin())
    line = section[centre - 16:centre + 17]
    paper = np.linspace(np.median(line[:4]), np.median(line[-4:]), len(line))
    kernel = np.maximum(paper - line, 0)
    if not kernel.sum():
        raise ValueError("No dark line found in the CAM01 calibration target")
    return kernel / kernel.sum(), centre


def fit_line(photo, axis, sharp=None, fixed_radius=None):
    """Fit blur radius and centre, or score a fixed radius on a line target."""
    section, guess = line_section(photo, axis)
    kernel = None
    if sharp is not None:
        kernel, guess = sharp
    half_window = 380
    position = np.arange(-half_window, half_window + 1)
    profile = section[guess - half_window:guess + half_window + 1]
    background = np.polynomial.legendre.legvander(position / half_window, 4)
    basis, _ = np.linalg.qr(background)
    residual = profile - basis @ (basis.T @ profile)
    energy = float(residual @ residual)

    def search(radii, centres):
        distances = position[None, :] - centres[:, None]
        best = (float("inf"), 0.0, 0.0)
        for radius in radii:
            profiles = disc_profile(distances, radius)
            if kernel is not None:
                profiles = np.array([np.convolve(row, kernel, mode="same") for row in profiles])
            projected = profiles - (profiles @ basis) @ basis.T
            inner = np.maximum(0.0, -(projected @ residual))
            norm = np.einsum("ij,ij->i", projected, projected)
            errors = energy - inner**2 / np.maximum(norm, 1e-30)
            index = int(errors.argmin())
            candidate = (float(errors[index]), float(radius), float(centres[index]))
            best = min(best, candidate)
        return best

    if fixed_radius is None:
        _, radius, centre = search(np.arange(0, 281, 2), np.arange(-100, 101, 2))
        fine = np.arange(max(0.5, radius - 3), min(280, radius + 3) + 0.1, 0.25)
        radii = np.r_[0.0, fine] if radius <= 3 else fine
    else:
        radii = [fixed_radius]
        _, _, centre = search(radii, np.arange(-100, 101, 2))
    error, radius, centre = search(radii, np.arange(centre - 3, centre + 3.1, 0.5))
    return {
        "radius": radius,
        "centre": guess + centre,
        "explained_fraction": max(0.0, 1 - error / energy) if energy else 0.0,
        "residual_rms": float(np.sqrt(max(0.0, error) / len(profile))),
    }


def check_dot(photo, x, y, radius):
    """Score three disc sizes on a separate black-dot calibration image."""
    x, y = round(x), round(y)
    half = max(24, round(1.5 * radius + 12))
    grid = np.arange(-half, half + 1, 2)
    xx, yy = np.meshgrid(grid, grid)
    patch = photo[y - half:y + half + 1:2, x - half:x + half + 1:2]
    if patch.shape != xx.shape:
        raise ValueError("Dot calibration window outside image")
    u, v = xx.ravel() / half, yy.ravel() / half
    design = np.column_stack((np.ones_like(u), u, v, u**2, u*v, v**2))
    basis, _ = np.linalg.qr(design)
    signal = patch.ravel() - basis @ (basis.T @ patch.ravel())
    energy = signal @ signal
    radii = np.maximum(0.5, radius * np.array([0.7, 1.0, 1.3]))
    best = np.zeros(3)
    for dx in (-12, -6, 0, 6, 12):
        for dy in (-12, -6, 0, 6, 12):
            distance = np.hypot(xx - dx, yy - dy).ravel()
            discs = np.clip((radii[:, None] - distance + 1) / 2, 0, 1)
            residual = discs - (discs @ basis) @ basis.T
            inner = np.maximum(0, -(residual @ signal))
            norm = np.sum(residual**2, axis=1)
            best = np.maximum(best, inner**2 / np.maximum(norm * energy, 1e-30))
    return {"radii": radii.tolist(), "explained_fraction": best.tolist()}
