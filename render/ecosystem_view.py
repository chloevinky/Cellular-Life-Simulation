"""Ecosystem visualization with biomes, plants, and HUD."""
from __future__ import annotations

import pygame
import numpy as np
from typing import Optional, Tuple

from plants.plant import PlantPopulation
from world.climate import Climate


class EcosystemView:
    """Renders the ecosystem with biomes, plants, and information overlay."""
    
    def __init__(self, width: int = 800, height: int = 800):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Miniature Earth - Ecosystem Simulation")
        
        # Fonts for HUD
        self.font_large = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        
        # Surface for biome background (updated less frequently)
        self.biome_surface = pygame.Surface((width, height))
        self.biome_update_counter = 0
        self.biome_update_interval = 300  # ticks between background recompute

        # Uniform shading overlay for day/night (fast)
        self.shade_surface = pygame.Surface((width, height))
        self.shade_surface.fill((0, 0, 0))
        
    def render(self, state):
        """Render the entire ecosystem."""
        # Recompute biome background infrequently for performance
        if self.biome_update_counter == 0 or (state.tick - self.biome_update_counter) >= self.biome_update_interval:
            self._render_biome_background(state.climate)
            self.biome_update_counter = state.tick

        # Draw static biome background
        self.screen.blit(self.biome_surface, (0, 0))

        # Apply global day/night shading (no per-cell cost)
        sun = getattr(state.climate, 'sun_intensity', 1.0)
        light_factor = 0.3 + 0.7 * float(sun)
        alpha = int(max(0, min(255, 255 * (1.0 - light_factor))))
        self.shade_surface.set_alpha(alpha)
        self.screen.blit(self.shade_surface, (0, 0))
        
        # Render plants
        self._render_plants(state.population, state.climate)
        
        # Render HUD
        self._render_hud(state)
        
        # Update display
        pygame.display.flip()
    
    def _render_biome_background(self, climate: Climate):
        """Render biome colors as background (vectorized + scaled)."""
        h, w = climate.height, climate.width
        temp = climate.temperature
        water = climate.water

        # Classify biomes (0=desert,1=grass,2=forest,3=tundra)
        biome = np.ones((h, w), dtype=np.uint8)
        mask_tundra = temp < 0.3
        mask_desert = (water < 0.3) & (~mask_tundra)
        mask_forest = (water > 0.6) & (~mask_tundra)
        biome[mask_tundra] = 3
        biome[mask_desert] = 0
        biome[mask_forest] = 2

        # Base colors palette
        palette = np.array([
            [194, 178, 128],  # Desert
            [141, 182, 104],  # Grassland
            [76, 132, 76],    # Forest
            [200, 200, 210],  # Tundra
        ], dtype=np.uint8)
        colors = palette[biome]

        # Subtle color variation (vectorized)
        temp_var = ((temp - 0.5) * 20).astype(np.int16)
        water_var = ((water - 0.5) * 10).astype(np.int16)
        r = np.clip(colors[..., 0].astype(np.int16) + temp_var, 0, 255).astype(np.uint8)
        g = np.clip(colors[..., 1].astype(np.int16) + water_var, 0, 255).astype(np.uint8)
        b = np.clip(colors[..., 2].astype(np.int16) - temp_var, 0, 255).astype(np.uint8)
        rgb = np.stack([r, g, b], axis=-1)

        # Convert to surface and scale to window size
        arr = np.swapaxes(rgb, 0, 1)  # to (w, h, 3) for surfarray
        surface_small = pygame.surfarray.make_surface(arr)
        self.biome_surface = pygame.transform.scale(surface_small, (self.width, self.height))
    
    def _render_plants(self, population: PlantPopulation, climate: Climate):
        """Render all plants as circles (lightweight)."""
        cell_h = self.height / climate.height
        cell_w = self.width / climate.width

        for plant in population.plants:
            if not plant.alive:
                continue
            screen_x = int(plant.x * cell_w + cell_w / 2)
            screen_y = int(plant.y * cell_h + cell_h / 2)
            base_radius = max(2, int(plant.size * 12))
            color = plant.get_color(population.species_colors)
            pygame.draw.circle(self.screen, color, (screen_x, screen_y), base_radius)
    
    
    def _render_hud(self, state):
        """Render heads-up display with statistics."""
        # Background for text
        hud_height = 80
        hud_bg = pygame.Surface((self.width, hud_height))
        hud_bg.set_alpha(200)
        hud_bg.fill((20, 20, 30))
        self.screen.blit(hud_bg, (0, 0))
        
        # Title
        title = self.font_large.render("Miniature Earth", True, (255, 255, 255))
        self.screen.blit(title, (10, 5))
        
        # Statistics
        y_offset = 35
        stats = [
            f"Population: {len([p for p in state.population.plants if p.alive])}",
            f"Species: {state.species_count}",
            f"Total Biomass: {state.total_energy:.1f}",
            f"Day: {state.tick // state.climate.day_length} | Time: {int(state.climate.day_phase * 24):02d}:00",
            f"Season: {'Spring' if state.climate.season_phase < 0.25 else 'Summer' if state.climate.season_phase < 0.5 else 'Fall' if state.climate.season_phase < 0.75 else 'Winter'}"
        ]
        
        for i, stat in enumerate(stats):
            text = self.font_small.render(stat, True, (200, 200, 200))
            self.screen.blit(text, (10, y_offset + i * 18))
        # Minimal HUD only; no controls or graphs
