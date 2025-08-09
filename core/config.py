from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


@dataclass
class WorldConfig:
    width: int
    height: int
    


@dataclass
class SimConfig:
    seed: int = 12345
    render_every: int = 2  # how many ticks per render
    max_plants: int = 4000
    stats_interval: int = 10
    initial_plants: int = 100
    reproduction_chance: float = 0.05  # per-tick prob when can_reproduce
    max_age: int = 1000


@dataclass
class ViewConfig:
    window_width: int = 640
    window_height: int = 640
    fps: int = 30
    title: str = "Cellular Life: Plants MVP"


@dataclass
class Config:
    world: WorldConfig
    sim: SimConfig
    view: ViewConfig
    plants: "PlantConfig"


@dataclass
class PlantInitConfig:
    temp_optimum: Tuple[float, float] = (0.0, 1.0)
    temp_tolerance: Tuple[float, float] = (0.1, 0.5)
    shade_tolerance: Tuple[float, float] = (0.0, 1.0)
    water_efficiency: Tuple[float, float] = (0.0, 1.0)
    growth_rate: Tuple[float, float] = (0.2, 0.8)
    photosynthesis_rate: Tuple[float, float] = (0.2, 0.8)
    maintenance_cost: Tuple[float, float] = (0.05, 0.2)
    max_size: Tuple[float, float] = (5.0, 20.0)
    repro_threshold: Tuple[float, float] = (3.0, 10.0)
    seed_dispersal: Tuple[float, float] = (1.0, 6.0)
    seed_energy: Tuple[float, float] = (0.5, 2.5)
    initial_energy: Tuple[float, float] = (1.0, 3.0)


@dataclass
class PlantMutationConfig:
    rate: float = 0.02
    sigma_traits: float = 0.1
    sigma_max_size: float = 1.0
    sigma_seed_energy: float = 0.2
    clip_traits: Tuple[float, float] = (0.01, 1.0)
    clip_max_size: Tuple[float, float] = (1.0, 50.0)
    clip_seed_energy: Tuple[float, float] = (0.5, 5.0)


@dataclass
class PlantConfig:
    init: PlantInitConfig = PlantInitConfig()
    mutation: PlantMutationConfig = PlantMutationConfig()
    physiology: "PlantPhysiologyConfig" = None
    species: "PlantSpeciesConfig" = None


@dataclass
class PlantPhysiologyConfig:
    photo_base_mult: float = 2.0
    nutrient_effect_mult: float = 2.0
    nutrient_cap: float = 1.0
    nutrient_consumption_base: float = 0.01
    water_use_base: float = 0.02
    shade_cast_coeff: float = 0.5
    shade_cap: float = 0.9
    maintenance_size_mult: float = 2.0
    water_eff_epsilon: float = 0.01
    low_light_threshold: float = 0.3
    nutrient_return_mult: float = 0.3


@dataclass
class PlantSpeciesConfig:
    distance_threshold: float = 0.5


def _world_from_dict(d: Dict[str, Any]) -> WorldConfig:
    return WorldConfig(
        width=int(d.get("width", 160)),
        height=int(d.get("height", 120)),
    )


def _sim_from_dict(d: Dict[str, Any]) -> SimConfig:
    return SimConfig(
        seed=int(d.get("seed", 12345)),
        render_every=int(d.get("render_every", 2)),
        max_plants=int(d.get("max_plants", 4000)),
        stats_interval=int(d.get("stats_interval", 10)),
        initial_plants=int(d.get("initial_plants", 100)),
        reproduction_chance=float(d.get("reproduction_chance", 0.05)),
        max_age=int(d.get("max_age", 1000)),
    )


def _view_from_dict(d: Dict[str, Any]) -> ViewConfig:
    return ViewConfig(
        window_width=int(d.get("window_width", 640)),
        window_height=int(d.get("window_height", 640)),
        fps=int(d.get("fps", 30)),
        title=str(d.get("title", "Cellular Life: Plants MVP")),
    )


def _tuple2(v: Any, default: Tuple[float, float]) -> Tuple[float, float]:
    try:
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return float(v[0]), float(v[1])
    except Exception:
        pass
    return default


def _plants_from_dict(d: Dict[str, Any]) -> PlantConfig:
    init_d = d.get("init", {}) or {}
    mut_d = d.get("mutation", {}) or {}
    phys_d = d.get("physiology", {}) or {}
    spec_d = d.get("species", {}) or {}
    init = PlantInitConfig(
        temp_optimum=_tuple2(init_d.get("temp_optimum"), (0.0, 1.0)),
        temp_tolerance=_tuple2(init_d.get("temp_tolerance"), (0.1, 0.5)),
        shade_tolerance=_tuple2(init_d.get("shade_tolerance"), (0.0, 1.0)),
        water_efficiency=_tuple2(init_d.get("water_efficiency"), (0.0, 1.0)),
        growth_rate=_tuple2(init_d.get("growth_rate"), (0.2, 0.8)),
        photosynthesis_rate=_tuple2(init_d.get("photosynthesis_rate"), (0.2, 0.8)),
        maintenance_cost=_tuple2(init_d.get("maintenance_cost"), (0.05, 0.2)),
        max_size=_tuple2(init_d.get("max_size"), (5.0, 20.0)),
        repro_threshold=_tuple2(init_d.get("repro_threshold"), (3.0, 10.0)),
        seed_dispersal=_tuple2(init_d.get("seed_dispersal"), (1.0, 6.0)),
        seed_energy=_tuple2(init_d.get("seed_energy"), (0.5, 2.5)),
        initial_energy=_tuple2(init_d.get("initial_energy"), (1.0, 3.0)),
    )
    mut = PlantMutationConfig(
        rate=float(mut_d.get("rate", 0.02)),
        sigma_traits=float(mut_d.get("sigma_traits", 0.1)),
        sigma_max_size=float(mut_d.get("sigma_max_size", 1.0)),
        sigma_seed_energy=float(mut_d.get("sigma_seed_energy", 0.2)),
        clip_traits=_tuple2(mut_d.get("clip_traits"), (0.01, 1.0)),
        clip_max_size=_tuple2(mut_d.get("clip_max_size"), (1.0, 50.0)),
        clip_seed_energy=_tuple2(mut_d.get("clip_seed_energy"), (0.5, 5.0)),
    )
    phys = PlantPhysiologyConfig(
        photo_base_mult=float(phys_d.get("photo_base_mult", 2.0)),
        nutrient_effect_mult=float(phys_d.get("nutrient_effect_mult", 2.0)),
        nutrient_cap=float(phys_d.get("nutrient_cap", 1.0)),
        nutrient_consumption_base=float(phys_d.get("nutrient_consumption_base", 0.01)),
        water_use_base=float(phys_d.get("water_use_base", 0.02)),
        shade_cast_coeff=float(phys_d.get("shade_cast_coeff", 0.5)),
        shade_cap=float(phys_d.get("shade_cap", 0.9)),
        maintenance_size_mult=float(phys_d.get("maintenance_size_mult", 2.0)),
        water_eff_epsilon=float(phys_d.get("water_eff_epsilon", 0.01)),
        low_light_threshold=float(phys_d.get("low_light_threshold", 0.3)),
        nutrient_return_mult=float(phys_d.get("nutrient_return_mult", 0.3)),
    )
    species = PlantSpeciesConfig(
        distance_threshold=float(spec_d.get("distance_threshold", 0.5)),
    )
    return PlantConfig(init=init, mutation=mut, physiology=phys, species=species)


def load_config(path: str | Path) -> Config:
    p = Path(path)
    if p.is_dir():
        p = p / "config" / "default.yaml"
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    world = _world_from_dict(raw.get("world", {}))
    sim = _sim_from_dict(raw.get("sim", {}))
    view = _view_from_dict(raw.get("view", {}))
    plants = _plants_from_dict(raw.get("plants", {}))
    return Config(world=world, sim=sim, view=view, plants=plants)
