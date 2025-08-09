from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class SeasonConfig:
    amplitude: float = 0.0
    period_steps: int = 0


@dataclass
class DisturbanceConfig:
    prob_per_step: float = 0.0
    patch_radius: int = 0
    magnitude: float = 1.0
    mode: str = "reset"  # reset | spike (not used yet)


@dataclass
class WorldConfig:
    width: int
    height: int
    torus: bool = True
    regrow_rate: float = 0.02
    diffusion_rate: float = 0.10
    max_resource: float = 1.0
    season: SeasonConfig = SeasonConfig()
    disturbance: DisturbanceConfig = DisturbanceConfig()


@dataclass
class SimConfig:
    steps: int = 100000
    render_every: int = 1
    log_every: int = 50
    seed: int = 12345


@dataclass
class ViewConfig:
    cell_size: int = 4
    fps: int = 60
    show_grid: bool = False
    title: str = "Cellular Life: Stage 1"


@dataclass
class Config:
    world: WorldConfig
    sim: SimConfig
    agents: "AgentsConfig"
    view: ViewConfig


def _season_from_dict(d: Dict[str, Any] | None) -> SeasonConfig:
    d = d or {}
    return SeasonConfig(
        amplitude=float(d.get("amplitude", 0.0)),
        period_steps=int(d.get("period_steps", 0)),
    )


def _disturbance_from_dict(d: Dict[str, Any] | None) -> DisturbanceConfig:
    d = d or {}
    return DisturbanceConfig(
        prob_per_step=float(d.get("prob_per_step", 0.0)),
        patch_radius=int(d.get("patch_radius", 0)),
        magnitude=float(d.get("magnitude", 1.0)),
        mode=str(d.get("mode", "reset")),
    )


def _world_from_dict(d: Dict[str, Any]) -> WorldConfig:
    season = _season_from_dict(d.get("season"))
    disturbance = _disturbance_from_dict(d.get("disturbance"))
    return WorldConfig(
        width=int(d.get("width", 160)),
        height=int(d.get("height", 120)),
        torus=bool(d.get("torus", True)),
        regrow_rate=float(d.get("regrow_rate", 0.02)),
        diffusion_rate=float(d.get("diffusion_rate", 0.10)),
        max_resource=float(d.get("max_resource", 1.0)),
        season=season,
        disturbance=disturbance,
    )


def _sim_from_dict(d: Dict[str, Any]) -> SimConfig:
    return SimConfig(
        steps=int(d.get("steps", 100000)),
        render_every=int(d.get("render_every", 1)),
        log_every=int(d.get("log_every", 50)),
        seed=int(d.get("seed", 12345)),
    )


def _view_from_dict(d: Dict[str, Any]) -> ViewConfig:
    return ViewConfig(
        cell_size=int(d.get("cell_size", 5)),
        fps=int(d.get("fps", 60)),
        show_grid=bool(d.get("show_grid", False)),
        title=str(d.get("title", "Cellular Life: Stage 1")),
    )


@dataclass
class AgentsConfig:
    init_population: int = 400
    max_population: int = 2000
    init_energy: float = 5.0
    metabolism: float = 0.1
    move_cost: float = 0.02
    graze_rate: float = 0.05
    graze_efficiency: float = 1.0
    spawn_attempts: int = 5


def _agents_from_dict(d: Dict[str, Any]) -> AgentsConfig:
    d = d or {}
    return AgentsConfig(
        init_population=int(d.get("init_population", 400)),
        max_population=int(d.get("max_population", 2000)),
        init_energy=float(d.get("init_energy", 5.0)),
        metabolism=float(d.get("metabolism", 0.1)),
        move_cost=float(d.get("move_cost", 0.02)),
        graze_rate=float(d.get("graze_rate", 0.05)),
        graze_efficiency=float(d.get("graze_efficiency", 1.0)),
        spawn_attempts=int(d.get("spawn_attempts", 5)),
    )


def load_config(path: str | Path) -> Config:
    p = Path(path)
    if p.is_dir():
        p = p / "config" / "default.yaml"
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    world = _world_from_dict(raw.get("world", {}))
    sim = _sim_from_dict(raw.get("sim", {}))
    agents = _agents_from_dict(raw.get("agents", {}))
    view = _view_from_dict(raw.get("view", {}))
    return Config(world=world, sim=sim, agents=agents, view=view)
