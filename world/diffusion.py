from __future__ import annotations

import numpy as np


def diffuse_toroidal(field: np.ndarray, rate: float) -> None:
    """In-place 2D diffusion with a simple 4-neighbor Laplacian on a torus.

    new = old + rate * (sum(neighbors) - 4 * old)
    """
    if rate <= 0.0:
        return
    up = np.roll(field, -1, axis=0)
    down = np.roll(field, 1, axis=0)
    left = np.roll(field, 1, axis=1)
    right = np.roll(field, -1, axis=1)
    field += rate * (up + down + left + right - 4.0 * field)
