from __future__ import annotations

from dataclasses import dataclass
from math import sin, tau
from typing import Tuple

import numpy as np

from core.config import WorldConfig
from utils.rng import RNG
from .diffusion import diffuse_toroidal


@dataclass
class ResourceWorld:
    cfg: WorldConfig
    rng: RNG

    def __post_init__(self) -> None:
        h, w = self.cfg.height, self.cfg.width
        # Start with small noise so the heatmap isn't uniform
        self.resource = (0.2 * self.rng.random((h, w))).astype(np.float32)
        self.t = 0  # steps elapsed

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.cfg.height, self.cfg.width)

    def season_factor(self) -> float:
        amp = self.cfg.season.amplitude
        period = self.cfg.season.period_steps
        if amp == 0.0 or period <= 0:
            return 1.0
        return 1.0 + amp * sin(tau * (self.t % period) / period)

    def regrow(self) -> None:
        # Logistic-like regrowth toward max_resource
        rate = self.cfg.regrow_rate * self.season_factor()
        if rate <= 0.0:
            return
        max_r = self.cfg.max_resource
        self.resource += rate * (max_r - self.resource)

    def disturb(self) -> None:
        dcfg = self.cfg.disturbance
        if dcfg.prob_per_step <= 0.0:
            return
        if float(self.rng.random()) >= dcfg.prob_per_step:
            return
        h, w = self.shape
        cy = int(self.rng.integers(0, h))
        cx = int(self.rng.integers(0, w))
        r = max(1, int(dcfg.patch_radius))
        yy, xx = np.ogrid[:h, :w]
        # Toroidal distance
        dy = np.minimum((yy - cy) % h, (cy - yy) % h)
        dx = np.minimum((xx - cx) % w, (cx - xx) % w)
        mask = (dy * dy + dx * dx) <= (r * r)
        if dcfg.mode == "reset":
            self.resource[mask] = 0.0
        else:  # spike
            self.resource[mask] += dcfg.magnitude
        # Clip after disturbance
        np.clip(self.resource, 0.0, self.cfg.max_resource, out=self.resource)

    def diffuse(self) -> None:
        diffuse_toroidal(self.resource, self.cfg.diffusion_rate)
        # Keep within bounds after diffusion
        np.clip(self.resource, 0.0, self.cfg.max_resource, out=self.resource)

    def step_env(self) -> None:
        self.regrow()
        self.diffuse()
        self.disturb()
        self.t += 1
