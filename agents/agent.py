from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

from core.config import AgentsConfig, WorldConfig
from utils.rng import RNG
from world.grid import Grid
from world.resources import ResourceWorld


@dataclass
class Agent:
    id: int
    y: int
    x: int
    energy: float
    genome: np.ndarray
    metabolism: float
    graze_efficiency: float
    repro_threshold: float
    color: Tuple[int, int, int]
    max_age: int = 0
    age: int = 0
    alive: bool = True

    def pos(self) -> Tuple[int, int]:
        return self.y, self.x


# 4-neighborhood + stay
_DIRS: List[Tuple[int, int]] = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]


def random_walk_move(agent: Agent, grid: Grid, world_cfg: WorldConfig, rng: RNG) -> Tuple[int, int]:
    dy, dx = _DIRS[int(rng.integers(0, len(_DIRS)))]
    ny, nx = agent.y + dy, agent.x + dx
    if world_cfg.torus:
        ny, nx = grid.wrap(ny, nx)
    else:
        ny = np.clip(ny, 0, grid.height - 1)
        nx = np.clip(nx, 0, grid.width - 1)
    return int(ny), int(nx)


def graze(world: ResourceWorld, y: int, x: int, rate: float, efficiency: float) -> float:
    available = float(world.resource[y, x])
    take = min(rate, available)
    if take <= 0.0:
        return 0.0
    world.resource[y, x] = available - take
    return take * efficiency


def step_agent(agent: Agent, grid: Grid, world: ResourceWorld, acfg: AgentsConfig, rng: RNG) -> None:
    if not agent.alive:
        return

    # Movement (random walk)
    ny, nx = random_walk_move(agent, grid, world.cfg, rng)
    moved = (ny != agent.y) or (nx != agent.x)
    if moved and grid.is_free(ny, nx):
        grid.move(agent.id, (agent.y, agent.x), (ny, nx))
        agent.y, agent.x = ny, nx
        agent.energy -= acfg.move_cost

    # Grazing at current cell
    gained = graze(world, agent.y, agent.x, acfg.graze_rate, agent.graze_efficiency)
    agent.energy += gained

    # Basal metabolism and aging
    agent.energy -= agent.metabolism
    agent.age += 1

    # Death check
    if agent.energy <= 0.0 or (agent.max_age > 0 and agent.age >= agent.max_age):
        agent.alive = False
        grid.remove(agent.y, agent.x)


def _scale_from_gene(g: float, span: float = 0.5) -> float:
    """Map a gene value to a multiplicative scale around 1.0 using tanh: [0.5, 1.5] when span=0.5."""
    return 1.0 + span * float(np.tanh(g))


def _color_from_genome(genome: np.ndarray) -> Tuple[int, int, int]:
    # Map first 3 genes via sigmoid to RGB
    g = 1.0 / (1.0 + np.exp(-genome[:3].astype(np.float64)))
    rgb = tuple(int(np.clip(v * 255.0, 0, 255)) for v in g)
    return rgb  # type: ignore


def _derive_traits_from_genome(genome: np.ndarray, acfg: AgentsConfig) -> Tuple[float, float, float, int, Tuple[int, int, int]]:
    m_scale = _scale_from_gene(float(genome[0]))
    ge_scale = _scale_from_gene(float(genome[1]))
    rt_scale = _scale_from_gene(float(genome[2]))
    metabolism = acfg.metabolism * m_scale
    graze_eff = acfg.graze_efficiency * ge_scale
    repro_thr = acfg.repro_threshold * rt_scale
    max_age = 0
    if acfg.max_age > 0:
        age_scale = _scale_from_gene(float(genome[3])) if genome.size > 3 else 1.0
        max_age = max(1, int(acfg.max_age * age_scale))
    color = _color_from_genome(genome)
    return metabolism, graze_eff, repro_thr, max_age, color


def _random_genome(acfg: AgentsConfig, rng: RNG) -> np.ndarray:
    # Small random normal around 0 so scales are ~1.0 initially
    return rng.normal(loc=0.0, scale=0.2, size=acfg.genome_size).astype(np.float32)


def spawn_initial_agents(acfg: AgentsConfig, grid: Grid, world: ResourceWorld, rng: RNG) -> list[Agent]:
    agents: list[Agent] = []
    next_id = 0
    attempts = max(1, int(acfg.spawn_attempts))
    for _ in range(acfg.init_population):
        spot = grid.random_free_cell(rng, attempts=attempts)
        if spot is None:
            break
        y, x = spot
        placed = grid.place(next_id, y, x)
        if not placed:
            continue
        genome = _random_genome(acfg, rng)
        metabolism, graze_eff, repro_thr, max_age, color = _derive_traits_from_genome(genome, acfg)
        agents.append(
            Agent(
                id=next_id,
                y=y,
                x=x,
                energy=float(acfg.init_energy),
                genome=genome,
                metabolism=metabolism,
                graze_efficiency=graze_eff,
                repro_threshold=repro_thr,
                color=color,
                max_age=max_age,
            )
        )
        next_id += 1
    return agents


def _adjacent_free_cell(grid: Grid, y: int, x: int, world_cfg: WorldConfig, rng: RNG) -> Optional[Tuple[int, int]]:
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    rng._rng.shuffle(dirs)
    for dy, dx in dirs:
        ny, nx = y + dy, x + dx
        if world_cfg.torus:
            ny, nx = grid.wrap(ny, nx)
        else:
            ny = int(np.clip(ny, 0, grid.height - 1))
            nx = int(np.clip(nx, 0, grid.width - 1))
        if grid.is_free(ny, nx):
            return ny, nx
    return None


def try_reproduce(agent: Agent, grid: Grid, world_cfg: WorldConfig, acfg: AgentsConfig, rng: RNG, next_id: int) -> Optional[Agent]:
    if not agent.alive:
        return None
    if agent.energy <= agent.repro_threshold:
        return None
    spot = _adjacent_free_cell(grid, agent.y, agent.x, world_cfg, rng)
    if spot is None:
        return None
    ny, nx = spot
    # Energy split
    child_energy = agent.energy * float(acfg.offspring_share)
    parent_energy = agent.energy - child_energy
    if child_energy <= 0.0 or parent_energy <= 0.0:
        return None
    # Mutate genome
    child_genome = (agent.genome + rng.normal(0.0, acfg.mutation_sigma, size=agent.genome.size)).astype(np.float32)
    metabolism, graze_eff, repro_thr, max_age, color = _derive_traits_from_genome(child_genome, acfg)
    if not grid.place(next_id, ny, nx):
        return None
    agent.energy = parent_energy
    child = Agent(
        id=next_id,
        y=ny,
        x=nx,
        energy=float(child_energy),
        genome=child_genome,
        metabolism=metabolism,
        graze_efficiency=graze_eff,
        repro_threshold=repro_thr,
        color=color,
        max_age=max_age,
    )
    return child
