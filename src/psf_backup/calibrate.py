"""Measure the disc radius of a defocus step from a line target (LSF_X or LSF_Y).

The target is a thin dark line on white paper. Averaged along the line, a photograph becomes a 1-D
cross-section: bright paper with a dip where the line is. A uniform disc PSF of radius r, integrated
along the line direction, is a half circle of width 2r:

    L_r(x) = 2 sqrt(r^2 - x^2) / (pi r^2)   for |x| <= r, 0 otherwise   (area 1)

so the blurred dip is the sharp dip convolved with L_r. The sharp camera's dip (the line's own
width and the camera's slight blur) is measured from the in-focus photograph of the same target.

Model of the blurred cross-section p(y) inside a window around the line:

    p(y) = background(y) - depth * g_r(y - y0),     g_r = k * L_r

where background is a low-order polynomial (vignetting makes the paper brightness vary by more than
the dip itself at large steps), k is the sharp line's absorption profile and y0 the line's position
in the blurred photograph (the two cameras are not perfectly aligned). For fixed (r, y0) the model is
linear in the background coefficients and the depth, so they are solved by least squares; r and y0
are found by grid search, first coarse, then fine.
"""
import numpy as np
import tifffile


def load_target(path, axis_along, margin=380):
    """Read a line-target photograph, oriented so the line runs along the columns.

    Args:
        path: TIFF file of an LSF target.
        axis_along: array axis that runs along the line (1 for the horizontal line of LSF_X,
            0 for the vertical line of LSF_Y).
        margin: pixels left out at both ends of the line, where vignetting is strongest.

    Returns:
        2-D float array: one column per position along the line, one row per position across it.
    """
    a = tifffile.imread(path).astype(np.float64)
    # keep the central part of the line; transpose a vertical line so it becomes horizontal
    return a[:, margin:-margin] if axis_along == 1 else a[margin:-margin, :].T


def straighten(a, tilt):
    """Shift each column so that a tilted line becomes level.

    Args:
        a: target array from load_target() (line along the columns).
        tilt: line position change per column (pixels per pixel), from line_tilt().

    Returns:
        Array of the same shape; column j is resampled at rows y + tilt * (j - centre).
    """
    if tilt == 0:
        return a
    rows = np.arange(a.shape[0], dtype=np.float64)
    centre = (a.shape[1] - 1) / 2
    out = np.empty_like(a)
    # linear interpolation along each column
    for j in range(a.shape[1]):
        out[:, j] = np.interp(rows + tilt * (j - centre), rows, a[:, j])
    return out


def cross_section(path, axis_along, margin=380, tilt=0.0):
    """Average a photograph of a line target along the line.

    Args:
        path: TIFF file of an LSF target.
        axis_along: array axis that runs along the line (1 for LSF_X, 0 for LSF_Y).
        margin: pixels left out at both ends of the line.
        tilt: line tilt to remove before averaging (pixels per pixel); 0 averages as it is.

    Returns:
        1-D float array: the mean pixel value at each position across the line.
    """
    return straighten(load_target(path, axis_along, margin), tilt).mean(axis=1)


def line_tilt(a, k, centre, r=None, segments=8):
    """Measure how much a line's position changes along its length.

    Args:
        a: target array from load_target().
        k: the sharp line's absorption profile (line_kernel()).
        centre: approximate line position across the line.
        r: disc radius of the photograph, or None for an in-focus photograph, whose dip is
            located by a parabola through its darkest pixels.
        segments: number of pieces the line is cut into.

    Returns:
        Tilt in pixels per pixel (slope of a straight line through the segment positions).
    """
    edges = np.linspace(0, a.shape[1], segments + 1).astype(int)
    xs, ys = [], []
    for i in range(segments):
        p = a[:, edges[i]:edges[i + 1]].mean(axis=1)
        if r is None:
            # sub-pixel minimum of the sharp dip
            j = centre - 20 + int(np.argmin(p[centre - 20:centre + 21]))
            y1, y2, y3 = p[j - 1], p[j], p[j + 1]
            ys.append(j + 0.5 * (y1 - y3) / (y1 - 2 * y2 + y3))
        else:
            # disc fit restricted to radii near the whole line's radius
            ys.append(fit_radius(p, k, centre, search=120, r_max=r + 3)["y0"])
        xs.append((edges[i] + edges[i + 1]) / 2)
    return float(np.polyfit(xs, ys, 1)[0])


def disc_lsf(r, offset=0.0):
    """Pixel-integrated half-circle line spread function of a uniform disc.

    Args:
        r: disc radius in pixels.
        offset: sub-pixel shift of the centre, in pixels.

    Returns:
        (values, start): values on consecutive pixels summing to 1, and the pixel index of the
        first value relative to the centre pixel.
    """
    n = int(np.ceil(r + abs(offset))) + 1
    x = np.arange(-n, n + 1, dtype=np.float64)

    def antiderivative(u):
        """Integral of 2 sqrt(r^2 - t^2) / (pi r^2) from 0 to u, with u clipped to [-r, r]."""
        u = np.clip(u, -r, r)
        return (u * np.sqrt(r * r - u * u) + r * r * np.arcsin(u / r)) / (np.pi * r * r)

    # exact integral over each pixel [x - 0.5, x + 0.5], shifted by the offset
    vals = antiderivative(x + 0.5 - offset) - antiderivative(x - 0.5 - offset)
    return vals / vals.sum(), -n


def line_kernel(sharp, half_width=30, degree=4, centre=None):
    """Absorption profile of the line in the in-focus photograph.

    Args:
        sharp: cross-section of the in-focus photograph (from cross_section()).
        half_width: rows on each side of the darkest point that belong to the line.
        degree: degree of the polynomial fitted to the paper around the line.
        centre: approximate index of the line; the darkest pixel within 50 pixels of it is used.
            None searches the whole cross-section.

    Returns:
        (k, centre): k is 1 - photo / paper on the 2 * half_width + 1 pixels around the line,
        clipped at 0; centre is the index of the darkest pixel.
    """
    if centre is None:
        c = int(np.argmin(sharp))
    else:
        # only look near the given position, so darker image edges are not mistaken for the line
        lo = max(int(centre) - 50, 0)
        c = lo + int(np.argmin(sharp[lo:int(centre) + 51]))
    y = np.arange(len(sharp), dtype=np.float64)
    # paper brightness from the pixels outside the line, within 300 pixels of it
    near = (np.abs(y - c) > half_width) & (np.abs(y - c) < 300)
    coef = np.polyfit((y[near] - c) / 300, sharp[near], degree)
    paper = np.polyval(coef, (y - c) / 300)
    k = 1 - sharp / paper
    return np.clip(k[c - half_width:c + half_width + 1], 0, None), c


def _polynomial_basis(n, degree):
    """Columns 1, t, t^2, ... on n points with t scaled to [-1, 1]."""
    t = np.linspace(-1, 1, n)
    return np.vander(t, degree + 1, increasing=True)


def fit_radius(blurred, k, centre_guess, search=150, window=380, degree=4,
               r_max=300.0, coarse_step=1.0, fine_step=0.05):
    """Find the disc radius that best explains a blurred cross-section.

    Args:
        blurred: cross-section of the defocused photograph.
        k: absorption profile of the line from line_kernel().
        centre_guess: expected index of the line in the blurred cross-section (the sharp
            photograph's line position).
        search: the line centre y0 is searched within centre_guess +- search pixels.
        window: half-width of the fitted window around centre_guess, in pixels; it must contain
            the whole blurred line and some paper on both sides.
        degree: degree of the background polynomial.
        r_max: largest radius tried.
        coarse_step: radius step of the coarse search.
        fine_step: radius step of the fine search around the coarse optimum.

    Returns:
        Dict with r (best radius), y0 (line centre), depth, ink (depth times the area of g
        divided by the paper level: about 1 when the blurred photograph keeps the line's total
        darkness), rms (residual), the curve of residual against radius from the coarse search,
        and the fitted model on the window.
    """
    lo, hi = centre_guess - window, centre_guess + window + 1
    p = blurred[lo:hi]
    n = len(p)
    # project out the background polynomial once; only the line term changes between candidates
    P = _polynomial_basis(n, degree)
    P_pinv = np.linalg.pinv(P)
    p_res = p - P @ (P_pinv @ p)
    base = float(p_res @ p_res)

    def evaluate(r, offset):
        """Best residual over all integer centres for one radius and sub-pixel offset."""
        lsf, start = disc_lsf(max(r, 1e-3), offset)
        g = np.convolve(k, lsf)                      # blurred line profile, unit-area kernel
        g_start = start - (len(k) - 1) // 2          # index of g[0] relative to the line centre
        shifts = np.arange(-search, search + 1)
        # one row per candidate centre: g placed at centre_guess + shift inside the window
        G = np.zeros((len(shifts), n))
        for i, s in enumerate(shifts):
            a = window + s + g_start
            b0, b1 = max(a, 0), min(a + len(g), n)
            if b1 > b0:
                G[i, b0:b1] = g[b0 - a:b1 - a]
        # residual after regressing p on [background, g] for every candidate at once
        G_res = G - (G @ P_pinv.T) @ P.T
        num = G_res @ p_res
        den = np.einsum("ij,ij->i", G_res, G_res)
        rss = base - num ** 2 / np.maximum(den, 1e-30)
        i = int(np.argmin(rss))
        return rss[i], shifts[i], -num[i] / den[i], G[i], g.sum()

    # coarse search over the radius, offset 0
    radii = np.arange(coarse_step, r_max + coarse_step, coarse_step)
    curve = np.array([evaluate(r, 0.0)[0] for r in radii])
    r0 = radii[int(np.argmin(curve))]
    # fine search over radius and sub-pixel offset around the coarse optimum
    best = None
    for r in np.arange(max(r0 - 2 * coarse_step, fine_step), r0 + 2 * coarse_step, fine_step):
        for off in (0.0, 0.25, 0.5, 0.75):
            res = evaluate(r, off)
            if best is None or res[0] < best[0][0]:
                best = (res, r, off)
    (rss, shift, depth, g_row, g_area), r, off = best
    # fitted background and model, for plotting and checks
    design = np.column_stack([P, -g_row])
    coef, *_ = np.linalg.lstsq(design, p, rcond=None)
    model = design @ coef
    paper = float(np.mean(P @ coef[:-1]))
    return {"r": float(r), "y0": float(centre_guess + shift + off), "depth": float(depth),
            "ink": float(depth * g_area / paper / k.sum()) if k.sum() > 0 else float("nan"),
            "rms": float(np.sqrt(rss / n)), "radii": radii, "curve": np.sqrt(curve / n),
            "window": (lo, hi), "data": p, "model": model, "background": P @ coef[:-1]}
