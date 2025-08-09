"""Plant agents with heritable traits and evolution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Any
import numpy as np

from utils.rng import RNG
from world.grid import Grid
from world.climate import Climate


@dataclass
class PlantGenome:
    """Heritable traits for plants."""
    # Environmental tolerances
    temp_optimum: float = 0.5      # Optimal temperature [0, 1]
    temp_tolerance: float = 0.3     # Temperature range tolerance
    shade_tolerance: float = 0.5    # Ability to grow in low light
    water_efficiency: float = 0.5    # Water use efficiency
    
    # Life history traits  
    growth_rate: float = 0.5        # How fast it grows
    max_size: float = 10.0          # Maximum energy/biomass
    repro_threshold: float = 5.0    # Energy needed to reproduce
    seed_dispersal: float = 3.0     # How far seeds spread
    seed_energy: float = 1.0        # Energy given to offspring
    
    # Metabolic traits
    photosynthesis_rate: float = 0.5
    maintenance_cost: float = 0.1
    
    # Evolution
    mutation_rate: float = 0.02     # Chance of mutation per trait
    
    def mutate(self, rng: RNG, mut_cfg: Any = None) -> PlantGenome:
        """Create mutated copy of genome using config-driven parameters."""
        new_genome = PlantGenome()

        rate = float(getattr(mut_cfg, 'rate', self.mutation_rate))
        sigma_traits = float(getattr(mut_cfg, 'sigma_traits', 0.1))
        sigma_max_size = float(getattr(mut_cfg, 'sigma_max_size', 1.0))
        sigma_seed_energy = float(getattr(mut_cfg, 'sigma_seed_energy', 0.2))
        clip_traits = getattr(mut_cfg, 'clip_traits', (0.01, 1.0))
        clip_max_size = getattr(mut_cfg, 'clip_max_size', (1.0, 50.0))
        clip_seed_energy = getattr(mut_cfg, 'clip_seed_energy', (0.5, 5.0))

        # Copy all traits with possible mutations
        for attr in ['temp_optimum', 'temp_tolerance', 'shade_tolerance',
                     'water_efficiency', 'growth_rate', 'photosynthesis_rate',
                     'maintenance_cost', 'repro_threshold', 'seed_dispersal']:
            value = getattr(self, attr)
            if rng.random() < rate:
                mutation = rng.normal(0, sigma_traits)
                value = float(np.clip(value + mutation, clip_traits[0], clip_traits[1]))
            setattr(new_genome, attr, value)

        # Mutate size traits
        if rng.random() < rate:
            new_genome.max_size = float(np.clip(self.max_size + rng.normal(0, sigma_max_size),
                                                clip_max_size[0], clip_max_size[1]))
        else:
            new_genome.max_size = self.max_size

        if rng.random() < rate:
            new_genome.seed_energy = float(np.clip(self.seed_energy + rng.normal(0, sigma_seed_energy),
                                                   clip_seed_energy[0], clip_seed_energy[1]))
        else:
            new_genome.seed_energy = self.seed_energy

        new_genome.mutation_rate = rate

        return new_genome
    
    def distance_to(self, other: PlantGenome) -> float:
        """Genetic distance to another genome (for species clustering)."""
        traits_self = np.array([
            self.temp_optimum, self.temp_tolerance, self.shade_tolerance,
            self.water_efficiency, self.growth_rate, self.photosynthesis_rate
        ])
        traits_other = np.array([
            other.temp_optimum, other.temp_tolerance, other.shade_tolerance,
            other.water_efficiency, other.growth_rate, other.photosynthesis_rate  
        ])
        return float(np.linalg.norm(traits_self - traits_other))


@dataclass
class Plant:
    """Individual plant agent."""
    id: int
    y: int
    x: int
    energy: float
    genome: PlantGenome
    species_id: int = 0
    age: int = 0
    alive: bool = True
    parent_id: Optional[int] = None
    
    # Cached computations
    shade_cast: float = 0.0  # How much shade this plant creates
    
    @property
    def size(self) -> float:
        """Plant size based on energy."""
        # Clamp energy and denominator to avoid negatives/zero
        energy = max(self.energy, 0.0)
        denom = max(self.genome.max_size, 1e-6)
        return float(min(energy / denom, 1.0))
    
    def get_color(self, species_colors: dict) -> Tuple[int, int, int]:
        """Get display color based on species."""
        if self.species_id not in species_colors:
            # Generate new species color
            np.random.seed(self.species_id)
            hue = np.random.random()
            # Convert HSV to RGB (simplified)
            r = int(128 + 127 * np.sin(2 * np.pi * hue))
            g = int(128 + 127 * np.sin(2 * np.pi * (hue + 0.33)))
            b = int(128 + 127 * np.sin(2 * np.pi * (hue + 0.67)))
            species_colors[self.species_id] = (r, g, b)
        return species_colors[self.species_id]
    
    def compute_fitness(self, climate: Climate, local_shade: float, phys_cfg: Any) -> float:
        """Compute fitness in current conditions (config-driven)."""
        # Temperature fitness (bell curve)
        temp = climate.temperature[self.y, self.x]
        temp_diff = abs(temp - self.genome.temp_optimum)
        tol = max(self.genome.temp_tolerance, 1e-6)
        temp_fitness = np.exp(-(temp_diff / tol) ** 2)
        
        # Light fitness (considering shade)
        light = climate.light[self.y, self.x] * (1 - local_shade)
        if light < float(getattr(phys_cfg, 'low_light_threshold', 0.3)):
            light_fitness = self.genome.shade_tolerance
        else:
            light_fitness = light
        
        # Water fitness
        water = climate.water[self.y, self.x]
        eps = float(getattr(phys_cfg, 'water_eff_epsilon', 0.01))
        water_fitness = min(1.0, water / (1 - self.genome.water_efficiency + eps))
        
        # Combined fitness
        return temp_fitness * light_fitness * water_fitness
    
    def photosynthesize(self, climate: Climate, local_shade: float, phys_cfg: Any) -> float:
        """Calculate energy gained from photosynthesis (config-driven)."""
        fitness = self.compute_fitness(climate, local_shade, phys_cfg)
        
        # Energy gain proportional to fitness and genome rate
        base_mult = float(getattr(phys_cfg, 'photo_base_mult', 2.0))
        base_gain = self.genome.photosynthesis_rate * base_mult
        
        # Size affects photosynthesis (bigger = more leaves). Use safe sqrt.
        size_factor = float(np.sqrt(max(self.size, 0.0)))
        
        # Nutrient limitation
        nutrients = climate.nutrients[self.y, self.x]
        nutrient_mult = float(getattr(phys_cfg, 'nutrient_effect_mult', 2.0))
        nutrient_cap = float(getattr(phys_cfg, 'nutrient_cap', 1.0))
        nutrient_factor = min(nutrient_cap, nutrients * nutrient_mult)
        
        energy_gain = base_gain * fitness * size_factor * nutrient_factor
        
        # Consume some nutrients
        consume_base = float(getattr(phys_cfg, 'nutrient_consumption_base', 0.01))
        consumed = min(consume_base * size_factor, nutrients)
        climate.nutrients[self.y, self.x] -= consumed
        
        # Consume water based on photosynthesis  
        water_use_base = float(getattr(phys_cfg, 'water_use_base', 0.02))
        water_use = water_use_base * (1 - self.genome.water_efficiency) * size_factor
        climate.water[self.y, self.x] = max(0, climate.water[self.y, self.x] - water_use)
        
        return energy_gain
    
    def metabolize(self, phys_cfg: Any) -> float:
        """Calculate energy cost of maintenance (config-driven)."""
        size_mult = float(getattr(phys_cfg, 'maintenance_size_mult', 2.0))
        return self.genome.maintenance_cost * (1 + self.size * size_mult)
    
    def can_reproduce(self) -> bool:
        """Check if plant has enough energy to reproduce."""
        return self.energy >= self.genome.repro_threshold
    
    def produce_seed(self, rng: RNG, next_id: int, init_cfg: Any = None, mut_cfg: Any = None) -> Plant:
        """Create offspring with mutated genome (config-driven)."""
        # Mutate genome
        child_genome = self.genome.mutate(rng, mut_cfg)
        
        # Determine seed location (dispersal)
        dispersal_dist = self.genome.seed_dispersal
        angle = rng.random() * 2 * np.pi
        dy = int(dispersal_dist * np.sin(angle))
        dx = int(dispersal_dist * np.cos(angle))
        
        # Pay energy cost
        self.energy -= self.genome.seed_energy
        
        # Create offspring
        child = Plant(
            id=next_id,
            y=self.y + dy,  # Will be wrapped/clipped by caller
            x=self.x + dx,
            energy=self.genome.seed_energy,
            genome=child_genome,
            species_id=self.species_id,  # Inherit species (may change later)
            parent_id=self.id
        )
        
        return child


class PlantPopulation:
    """Manages all plants and species clustering."""
    
    def __init__(self, distance_threshold: float = 0.5):
        self.plants: List[Plant] = []
        # Fast lookup from grid occupant id -> Plant
        self.id_index: dict[int, Plant] = {}
        self.next_species_id = 0
        self.species_colors: dict = {}
        self.species_centers: dict = {}  # species_id -> representative genome
        self.distance_threshold = float(distance_threshold)
        
    def add_plant(self, plant: Plant):
        """Add a new plant to population."""
        # Assign species based on genetic similarity
        if not self.species_centers:
            # First plant creates first species
            plant.species_id = self.next_species_id
            self.species_centers[self.next_species_id] = plant.genome
            self.next_species_id += 1
        else:
            # Find closest species
            min_dist = float('inf')
            closest_species = 0
            
            for species_id, center_genome in self.species_centers.items():
                dist = plant.genome.distance_to(center_genome)
                if dist < min_dist:
                    min_dist = dist
                    closest_species = species_id
            
            # If very different, create new species
            if min_dist > self.distance_threshold:  # Threshold for new species
                plant.species_id = self.next_species_id
                self.species_centers[self.next_species_id] = plant.genome
                self.next_species_id += 1
            else:
                plant.species_id = closest_species
        
        self.plants.append(plant)
        self.id_index[plant.id] = plant
    
    def remove_dead(self):
        """Remove dead plants."""
        self.plants = [p for p in self.plants if p.alive]
        # Rebuild index to drop dead plants
        self.id_index = {p.id: p for p in self.plants}
    
    def get_local_shade(self, y: int, x: int, grid: Grid, phys_cfg: Any) -> float:
        """Calculate shade at position from nearby plants (config-driven)."""
        total_shade = 0.0
        coeff = float(getattr(phys_cfg, 'shade_cast_coeff', 0.5))
        shade_cap = float(getattr(phys_cfg, 'shade_cap', 0.9))
        
        # Check 3x3 neighborhood
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny = (y + dy) % grid.height
                nx = (x + dx) % grid.width
                
                occupant_id = grid.occ[ny, nx]
                if occupant_id != -1:
                    # O(1) lookup of neighboring plant
                    plant = self.id_index.get(occupant_id)
                    if plant and plant.alive:
                        # Larger plants cast more shade
                        shade = plant.size * coeff
                        # Shade decreases with distance
                        dist = abs(dy) + abs(dx)
                        if dist > 0:
                            shade /= dist
                        total_shade += shade
        
        return min(total_shade, shade_cap)
