"""Climate system with light, water, temperature fields and biomes."""
from __future__ import annotations

from dataclasses import dataclass
from math import sin, cos, pi
import numpy as np
from typing import Tuple

from utils.rng import RNG


@dataclass
class Climate:
    """Manages environmental fields: light, water, temperature."""
    height: int
    width: int
    rng: RNG
    
    # Time tracking
    tick: int = 0
    day_length: int = 60  # ticks per day
    season_length: int = 2000  # ticks per season cycle
    
    def __post_init__(self):
        h, w = self.height, self.width
        
        # Initialize fields
        self.light = np.ones((h, w), dtype=np.float32)
        self.water = np.ones((h, w), dtype=np.float32) * 0.5
        self.nutrients = np.ones((h, w), dtype=np.float32) * 0.5
        self.temperature = np.ones((h, w), dtype=np.float32) * 0.5
        
        # Create latitude gradient for temperature (colder at top/bottom)
        lat_gradient = np.linspace(0, pi, h)
        self.base_temp = np.sin(lat_gradient)[:, np.newaxis] * 0.7 + 0.3
        self.base_temp = self.base_temp.astype(np.float32)
        # Broadcast base temperature across width for full 2D field
        self.base_temp_full = np.repeat(self.base_temp, w, axis=1)
        # Initialize temperature field from base
        self.temperature = self.base_temp_full.copy()
        
        # Create moisture zones (bands with some noise)
        moisture_bands = np.sin(np.linspace(0, 3*pi, h))[:, np.newaxis]
        noise = self.rng.random((h, w)) * 0.3 - 0.15
        self.moisture_zones = np.clip(moisture_bands * 0.5 + 0.5 + noise, 0.2, 0.9).astype(np.float32)
        
        # Initialize water with moisture zones
        self.water = self.moisture_zones.copy()
        
        # Nutrient patches
        self.nutrients = self.rng.random((h, w)).astype(np.float32) * 0.5 + 0.3
        
    @property
    def day_phase(self) -> float:
        """Current phase in day cycle [0, 1]."""
        return (self.tick % self.day_length) / self.day_length
    
    @property
    def season_phase(self) -> float:
        """Current phase in seasonal cycle [0, 1]."""
        return (self.tick % self.season_length) / self.season_length
    
    def get_biome(self, y: int, x: int) -> int:
        """Get biome type at position: 0=desert, 1=grassland, 2=forest, 3=tundra."""
        temp = self.temperature[y, x]
        water = self.water[y, x]
        
        if temp < 0.3:  # Cold
            return 3  # Tundra
        elif water < 0.3:  # Dry
            return 0  # Desert
        elif water > 0.6:  # Wet
            return 2  # Forest
        else:
            return 1  # Grassland
    
    def get_biome_color(self, y: int, x: int) -> Tuple[int, int, int]:
        """Get RGB color for biome at position."""
        biome = self.get_biome(y, x)
        # Base colors for each biome
        colors = [
            (194, 178, 128),  # Desert - sandy
            (141, 182, 104),  # Grassland - green
            (76, 132, 76),    # Forest - dark green
            (200, 200, 210),  # Tundra - pale blue-gray
        ]
        base_r, base_g, base_b = colors[biome]
        
        # Add subtle variation based on exact values
        temp_var = int((self.temperature[y, x] - 0.5) * 20)
        water_var = int((self.water[y, x] - 0.5) * 10)
        
        r = np.clip(base_r + temp_var, 0, 255)
        g = np.clip(base_g + water_var, 0, 255)
        b = np.clip(base_b - temp_var, 0, 255)
        
        return int(r), int(g), int(b)
    
    def step(self):
        """Update climate fields for one tick."""
        self.tick += 1
        
        # Day/night cycle for light
        self.sun_intensity = 0.5 + 0.5 * sin(2 * pi * self.day_phase)
        self.light.fill(self.sun_intensity)
        
        # Add dawn/dusk gradient
        if 0.2 < self.day_phase < 0.3 or 0.7 < self.day_phase < 0.8:
            gradient = np.linspace(0.3, 0.7, self.width)
            self.light += gradient[np.newaxis, :] * 0.2
        
        # Seasonal temperature variation (apply to full 2D field)
        season_temp = 0.2 * sin(2 * pi * self.season_phase)
        self.temperature = (self.base_temp_full + season_temp).astype(np.float32)
        
        # Water dynamics - slow regeneration toward moisture zones
        regen_rate = 0.01
        self.water += (self.moisture_zones - self.water) * regen_rate
        
        # Rainfall events (random patches)
        if self.rng.random() < 0.05:  # 5% chance per tick
            cy = int(self.rng.integers(0, self.height))
            cx = int(self.rng.integers(0, self.width))
            radius = 10
            
            # Create rainfall patch
            y_indices, x_indices = np.ogrid[:self.height, :self.width]
            dist_sq = (y_indices - cy)**2 + (x_indices - cx)**2
            rain_mask = dist_sq <= radius**2
            self.water[rain_mask] = np.minimum(self.water[rain_mask] + 0.3, 1.0)
        
        # Nutrient regeneration (very slow)
        self.nutrients += 0.001
        self.nutrients = np.clip(self.nutrients, 0, 1)
        
        # Simple water diffusion
        self._diffuse_field(self.water, 0.05)
        
        # Ensure all fields stay in valid ranges
        self.light = np.clip(self.light, 0, 1)
        self.water = np.clip(self.water, 0, 1)
        self.temperature = np.clip(self.temperature, 0, 1)
    
    def _diffuse_field(self, field: np.ndarray, rate: float):
        """Simple diffusion for a field."""
        if rate <= 0:
            return
        # Faster 4-neighbor diffusion using roll (toroidal)
        up = np.roll(field, -1, axis=0)
        down = np.roll(field, 1, axis=0)
        left = np.roll(field, 1, axis=1)
        right = np.roll(field, -1, axis=1)
        neighbors_sum = up + down + left + right
        field[:, :] = (1 - rate) * field + (rate * 0.25) * neighbors_sum
