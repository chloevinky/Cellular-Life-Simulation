from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

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
    gained = graze(world, agent.y, agent.x, acfg.graze_rate, acfg.graze_efficiency)
    agent.energy += gained

    # Basal metabolism and aging
    agent.energy -= acfg.metabolism
    agent.age += 1

    # Death check
    if agent.energy <= 0.0:
        agent.alive = False
        grid.remove(agent.y, agent.x)


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
        agents.append(Agent(id=next_id, y=y, x=x, energy=float(acfg.init_energy)))
        next_id += 1
    return agents
