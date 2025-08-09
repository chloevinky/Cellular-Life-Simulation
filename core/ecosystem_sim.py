"""Main ecosystem simulation with plants, climate, and evolution."""
from __future__ import annotations

import pygame
from dataclasses import dataclass
import numpy as np
from pathlib import Path
from typing import Optional

from core.config import Config, load_config
from render.ecosystem_view import EcosystemView
from utils.rng import RNG
from world.climate import Climate
from world.grid import Grid
from plants.plant import Plant, PlantGenome, PlantPopulation


@dataclass
class EcosystemState:
    """State of the entire ecosystem simulation."""
    cfg: Config
    rng: RNG
    climate: Climate
    grid: Grid
    population: PlantPopulation
    view: EcosystemView
    
    tick: int = 0
    next_plant_id: int = 0
    paused: bool = False
    speed_multiplier: int = 1
    
    max_plants: int = 4000
    stats_interval: int = 10
    
    # Statistics
    total_energy: float = 0.0
    species_count: int = 0
    

def create_initial_plants(state: EcosystemState, num_plants: int = 100):
    """Seed initial plant population."""
    for _ in range(num_plants):
        # Random position
        y = state.rng.integers(0, state.climate.height)
        x = state.rng.integers(0, state.climate.width)
        
        # Check if position is free
        if state.grid.is_free(y, x):
            # Create random genome sampled from config ranges
            genome = PlantGenome()
            init = state.cfg.plants.init
            def uni(lo, hi):
                return float(lo + (hi - lo) * state.rng.random())
            
            genome.temp_optimum = uni(*init.temp_optimum)
            genome.temp_tolerance = uni(*init.temp_tolerance)
            genome.shade_tolerance = uni(*init.shade_tolerance)
            genome.water_efficiency = uni(*init.water_efficiency)
            genome.growth_rate = uni(*init.growth_rate)
            genome.photosynthesis_rate = uni(*init.photosynthesis_rate)
            genome.maintenance_cost = uni(*init.maintenance_cost)
            genome.max_size = uni(*init.max_size)
            genome.repro_threshold = uni(*init.repro_threshold)
            genome.seed_dispersal = uni(*init.seed_dispersal)
            genome.seed_energy = uni(*init.seed_energy)
            
            # Create plant
            plant = Plant(
                id=state.next_plant_id,
                y=y,
                x=x,
                energy=uni(*init.initial_energy),
                genome=genome
            )
            
            state.population.add_plant(plant)
            state.grid.place(plant.id, y, x)
            state.next_plant_id += 1


def step_ecosystem(state: EcosystemState):
    """Update ecosystem for one tick."""
    if state.paused:
        return
    
    # Multiple steps if speed > 1
    for _ in range(state.speed_multiplier):
        # Update climate
        state.climate.step()
        
        # Step each plant
        new_plants = []
        for plant in state.population.plants:
            if not plant.alive:
                continue
            
            # Calculate local shade from neighbors
            local_shade = state.population.get_local_shade(
                plant.y, plant.x, state.grid, state.cfg.plants.physiology
            )
            
            # Photosynthesis
            energy_gained = plant.photosynthesize(
                state.climate, local_shade, state.cfg.plants.physiology
            )
            plant.energy += energy_gained
            
            # Metabolism
            energy_cost = plant.metabolize(state.cfg.plants.physiology)
            plant.energy -= energy_cost
            
            # Growth (implicit through energy)
            
            # Age
            plant.age += 1
            
            # Death check (age cap optional via config)
            max_age = int(state.cfg.sim.max_age)
            age_out = (max_age > 0 and plant.age > max_age)
            if plant.energy <= 0 or age_out:
                plant.alive = False
                state.grid.remove(plant.y, plant.x)
                # Return nutrients to soil
                ret_mult = float(getattr(state.cfg.plants.physiology, 'nutrient_return_mult', 0.3))
                state.climate.nutrients[plant.y, plant.x] = min(
                    1.0,
                    state.climate.nutrients[plant.y, plant.x] + plant.size * ret_mult
                )
                continue
            
            # Reproduction (gated by population cap)
            if (
                plant.can_reproduce()
                and len(state.population.plants) < state.max_plants
                and state.rng.random() < float(state.cfg.sim.reproduction_chance)
            ):
                seed = plant.produce_seed(
                    state.rng,
                    state.next_plant_id,
                    state.cfg.plants.init,
                    state.cfg.plants.mutation,
                )
                
                # Wrap/clip position
                seed.y = seed.y % state.climate.height
                seed.x = seed.x % state.climate.width
                
                # Try to place seed
                if state.grid.is_free(seed.y, seed.x):
                    new_plants.append(seed)
                    state.grid.place(seed.id, seed.y, seed.x)
                    state.next_plant_id += 1
        
        # Add new plants to population
        for plant in new_plants:
            state.population.add_plant(plant)
        
        # Remove dead plants
        state.population.remove_dead()
        
        # Update statistics less frequently
        if state.tick % state.stats_interval == 0:
            state.total_energy = sum(p.energy for p in state.population.plants if p.alive)
            state.species_count = len({p.species_id for p in state.population.plants if p.alive})
        
        state.tick += 1


 


def run_ecosystem(cfg: Config) -> None:
    """Main ecosystem simulation loop."""
    # Initialize components
    rng = RNG(seed=cfg.sim.seed)
    
    # Use world dimensions from config
    height = cfg.world.height
    width = cfg.world.width
    
    climate = Climate(height, width, rng)
    grid = Grid(height, width)
    population = PlantPopulation(distance_threshold=cfg.plants.species.distance_threshold)
    
    # Create view
    view = EcosystemView(width=cfg.view.window_width, height=cfg.view.window_height)
    
    # Create state
    state = EcosystemState(
        cfg=cfg,
        rng=rng,
        climate=climate,
        grid=grid,
        population=population,
        view=view
    )
    # Apply simulation tunables from config
    state.max_plants = int(cfg.sim.max_plants)
    state.stats_interval = int(cfg.sim.stats_interval)
    
    # Create initial plants
    create_initial_plants(state, num_plants=int(cfg.sim.initial_plants))
    
    clock = pygame.time.Clock()
    running = True
    # Render only every N ticks to reduce draw cost (from config)
    render_every = max(1, int(cfg.sim.render_every))
    
    while running:
        # Autonomous mode: only handle quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Update simulation
        step_ecosystem(state)
        
        # Render less frequently for performance
        if state.tick % render_every == 0:
            state.view.render(state)
        
        # Control frame rate from config
        clock.tick(int(cfg.view.fps))
    
    pygame.quit()


def run_with_config_path(config_path: Optional[str | Path] = None) -> None:
    """Run ecosystem with config."""
    if config_path is None:
        cfg = load_config(Path(__file__).resolve().parents[1])
    else:
        cfg = load_config(config_path)
    run_ecosystem(cfg)
