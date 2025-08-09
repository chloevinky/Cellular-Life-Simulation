from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.config import Config, load_config
from render.pygame_view import PygameView
from utils.rng import RNG
from world.resources import ResourceWorld
from world.grid import Grid
from agents.agent import Agent, spawn_initial_agents, step_agent, try_reproduce


@dataclass
class State:
    cfg: Config
    rng: RNG
    world: ResourceWorld
    view: PygameView
    grid: Grid
    agents: list[Agent]
    next_id: int
    t: int = 0


def make_state(cfg: Config) -> State:
    rng = RNG(seed=cfg.sim.seed)
    world = ResourceWorld(cfg=cfg.world, rng=rng)
    view = PygameView(view_cfg=cfg.view, world_cfg=cfg.world)
    grid = Grid(height=cfg.world.height, width=cfg.world.width)
    agents = spawn_initial_agents(cfg.agents, grid, world, rng)
    next_id = (max((a.id for a in agents), default=-1) + 1) if agents else 0
    return State(cfg=cfg, rng=rng, world=world, view=view, grid=grid, agents=agents, next_id=next_id)


def run(cfg: Config) -> None:
    state = make_state(cfg)

    try:
        for step in range(cfg.sim.steps):
            state.view.process_events()
            if state.view.closed:
                break

            state.world.step_env()

            # Step agents
            for a in state.agents:
                if a.alive:
                    step_agent(a, state.grid, state.world, cfg.agents, state.rng)
            # Compact dead agents BEFORE reproduction so capacity uses live count
            state.agents = [a for a in state.agents if a.alive]

            # Reproduction (Stage 3)
            newborns: list[Agent] = []
            alive_agents = state.agents  # already compacted
            capacity_left = max(0, cfg.agents.max_population - len(alive_agents))
            if capacity_left > 0:
                for a in alive_agents:
                    if len(newborns) >= capacity_left:
                        break
                    child = try_reproduce(a, state.grid, cfg.world, cfg.agents, state.rng, state.next_id)
                    if child is not None:
                        newborns.append(child)
                        state.next_id += 1
                        if len(newborns) >= capacity_left:
                            break
            if newborns:
                state.agents.extend(newborns)

            fps_val = state.view.tick()
            # Prepare positions and colors for rendering to avoid coupling render to Agent type
            alive_agents = [a for a in state.agents if a.alive]
            positions = [(a.y, a.x) for a in alive_agents]
            colors = [a.color for a in alive_agents]
            state.view.render(state.world.resource, cfg.world.max_resource, positions, colors=colors, fps_val=fps_val, step=step, pop=len(alive_agents))
            state.t += 1
    finally:
        state.view.quit()


def run_with_config_path(config_path: Optional[str | Path] = None) -> None:
    if config_path is None:
        # project root
        cfg = load_config(Path(__file__).resolve().parents[1])
    else:
        cfg = load_config(config_path)
    run(cfg)
