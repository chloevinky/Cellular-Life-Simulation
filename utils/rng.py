from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RNG:
    seed: int

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def integers(self, low: int, high: Optional[int] = None, size=None) -> np.ndarray:
        return self._rng.integers(low, high=high, size=size)

    def random(self, size=None) -> np.ndarray:
        return self._rng.random(size=size)

    def choice(self, a, size=None, replace=True, p=None):
        return self._rng.choice(a, size=size, replace=replace, p=p)

    def normal(self, loc=0.0, scale=1.0, size=None) -> np.ndarray:
        return self._rng.normal(loc=loc, scale=scale, size=size)

    def get_state(self):
        return self._rng.bit_generator.state

    def set_state(self, state) -> None:
        self._rng.bit_generator.state = state
