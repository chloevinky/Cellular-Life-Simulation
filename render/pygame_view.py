from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pygame

from core.config import ViewConfig, WorldConfig


def _colormap(norm: np.ndarray) -> np.ndarray:
    """Map [0,1] float array to RGB uint8 using a simple blue→green→yellow→red gradient."""
    x = np.clip(norm, 0.0, 1.0)
    # piecewise gradient: 0-0.33 blue->green, 0.33-0.66 green->yellow, 0.66-1 yellow->red
    r = np.zeros_like(x)
    g = np.zeros_like(x)
    b = np.zeros_like(x)

    # blue to green
    m1 = x <= 1/3
    t1 = np.zeros_like(x)
    t1[m1] = x[m1] * 3.0
    r[m1] = 0
    g[m1] = t1[m1]
    b[m1] = 1 - t1[m1]

    # green to yellow
    m2 = (x > 1/3) & (x <= 2/3)
    t2 = np.zeros_like(x)
    t2[m2] = (x[m2] - 1/3) * 3.0
    r[m2] = t2[m2]
    g[m2] = 1.0
    b[m2] = 0.0

    # yellow to red
    m3 = x > 2/3
    t3 = np.zeros_like(x)
    t3[m3] = (x[m3] - 2/3) * 3.0
    r[m3] = 1.0
    g[m3] = 1.0 - t3[m3]
    b[m3] = 0.0

    rgb = np.stack([(r * 255).astype(np.uint8), (g * 255).astype(np.uint8), (b * 255).astype(np.uint8)], axis=-1)
    return rgb


@dataclass
class PygameView:
    view_cfg: ViewConfig
    world_cfg: WorldConfig

    def __post_init__(self) -> None:
        pygame.init()
        self.cell = max(1, int(self.view_cfg.cell_size))
        self.size_px = (self.world_cfg.width * self.cell, self.world_cfg.height * self.cell)
        self.screen = pygame.display.set_mode(self.size_px)
        pygame.display.set_caption(self.view_cfg.title)
        self.clock = pygame.time.Clock()
        self.closed = False
        self.font: Optional[pygame.font.Font] = None
        try:
            self.font = pygame.font.SysFont("consolas", 14)
        except Exception:
            self.font = None

    def process_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.closed = True
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.closed = True

    def _draw_resource(self, resource: np.ndarray, max_resource: float) -> None:
        norm = (resource / max_resource).astype(np.float32)
        rgb = _colormap(norm)
        surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))  # pygame expects (w,h,3)
        if self.cell != 1:
            surf = pygame.transform.scale(surf, self.size_px)
        self.screen.blit(surf, (0, 0))

    def _draw_agents(self, positions: list[tuple[int, int]]) -> None:
        if not positions:
            return
        cs = self.cell
        color = (255, 255, 255)
        if cs <= 2:
            # draw as single pixels (scaled surface already)
            for (y, x) in positions:
                self.screen.set_at((x * cs, y * cs), color)
        else:
            for (y, x) in positions:
                rect = pygame.Rect(x * cs, y * cs, cs, cs)
                pygame.draw.rect(self.screen, color, rect)

    def _draw_hud(self, fps_val: Optional[float], step: Optional[int]) -> None:
        if self.font is None:
            return
        hud_lines = []
        if step is not None:
            hud_lines.append(f"step: {step}")
        if fps_val is not None:
            hud_lines.append(f"fps: {fps_val:.1f}")
        if hud_lines:
            hud_text = "  ".join(hud_lines)
            text_surf = self.font.render(hud_text, True, (255, 255, 255))
            self.screen.blit(text_surf, (8, 8))

    def render_resource(self, resource: np.ndarray, max_resource: float, fps_val: Optional[float] = None, step: Optional[int] = None) -> None:
        # Backward-compatible: just draw resource and HUD
        self._draw_resource(resource, max_resource)
        self._draw_hud(fps_val, step)
        pygame.display.flip()

    def render(self, resource: np.ndarray, max_resource: float, positions: list[tuple[int, int]], fps_val: Optional[float] = None, step: Optional[int] = None) -> None:
        self._draw_resource(resource, max_resource)
        self._draw_agents(positions)
        self._draw_hud(fps_val, step)
        pygame.display.flip()

    def tick(self) -> float:
        self.clock.tick(self.view_cfg.fps)
        return self.clock.get_fps() or 0.0

    def quit(self) -> None:
        pygame.quit()
