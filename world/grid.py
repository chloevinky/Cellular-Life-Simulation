from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class Grid:
    height: int
    width: int

    def __post_init__(self) -> None:
        self.occ = np.full((self.height, self.width), -1, dtype=np.int32)

    def in_bounds(self, y: int, x: int) -> bool:
        return 0 <= y < self.height and 0 <= x < self.width

    def wrap(self, y: int, x: int) -> Tuple[int, int]:
        return y % self.height, x % self.width

    def is_free(self, y: int, x: int) -> bool:
        return self.occ[y, x] == -1

    def place(self, agent_id: int, y: int, x: int) -> bool:
        if self.occ[y, x] != -1:
            return False
        self.occ[y, x] = agent_id
        return True

    def move(self, agent_id: int, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
        fy, fx = from_pos
        ty, tx = to_pos
        if self.occ[fy, fx] != agent_id:
            return False
        if self.occ[ty, tx] != -1:
            return False
        self.occ[fy, fx] = -1
        self.occ[ty, tx] = agent_id
        return True

    def remove(self, y: int, x: int) -> None:
        self.occ[y, x] = -1

    def random_free_cell(self, rng, attempts: int = 10) -> Optional[Tuple[int, int]]:
        # Try a few random samples; if none free, scan fallback
        h, w = self.height, self.width
        for _ in range(attempts):
            y = int(rng.integers(0, h))
            x = int(rng.integers(0, w))
            if self.occ[y, x] == -1:
                return y, x
        # fallback scan
        ys = np.arange(h); xs = np.arange(w)
        rng._rng.shuffle(ys)
        rng._rng.shuffle(xs)
        for y in ys:
            for x in xs:
                if self.occ[y, x] == -1:
                    return int(y), int(x)
        return None
