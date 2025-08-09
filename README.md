# Cellular Life Simulation (Autonomous Plants-Only MVP)

An autonomous, miniature-Earth ecosystem simulation focusing on plants that grow, compete, and evolve under changing climate fields.

## Quick Start

Prereqs: Python 3.9

```bash
pip install -r requirements.txt
python main.py --config .
```

This launches a Pygame window. The simulation runs with no user input; close the window to exit.

## Configuration

Default config: `config/default.yaml`

Minimal schema used by the plants-only MVP:

```yaml
world:
  width: 200         # grid width in cells
  height: 150        # grid height in cells

sim:
  seed: 12345        # RNG seed for reproducibility
  render_every: 2    # render once every N simulation ticks
  max_plants: 4000   # population cap to stabilize performance
  stats_interval: 10 # recalc/print stats once every N ticks
  initial_plants: 100       # plants to seed at start
  reproduction_chance: 0.05 # per-tick probability when plant can_reproduce()
  max_age: 1000             # ticks; set 0 to disable age-based death

plants:
  init:                      # trait ranges for initial genomes [lo, hi]
    temp_optimum: [0.0, 1.0]
    temp_tolerance: [0.1, 0.5]
    shade_tolerance: [0.0, 1.0]
    water_efficiency: [0.0, 1.0]
    growth_rate: [0.2, 0.8]
    photosynthesis_rate: [0.2, 0.8]
    maintenance_cost: [0.05, 0.2]
    max_size: [5.0, 20.0]
    repro_threshold: [3.0, 10.0]
    seed_dispersal: [1.0, 6.0]
    seed_energy: [0.5, 2.5]
    initial_energy: [1.0, 3.0]

  mutation:                  # mutation behavior and clipping
    rate: 0.02
    sigma_traits: 0.1
    sigma_max_size: 1.0
    sigma_seed_energy: 0.2
    clip_traits: [0.01, 1.0]
    clip_max_size: [1.0, 50.0]
    clip_seed_energy: [0.5, 5.0]

  physiology:                # energy flow constants
    photo_base_mult: 2.0
    nutrient_effect_mult: 2.0
    nutrient_cap: 1.0
    nutrient_consumption_base: 0.01
    water_use_base: 0.02
    shade_cast_coeff: 0.5
    shade_cap: 0.9
    maintenance_size_mult: 2.0
    water_eff_epsilon: 0.01
    low_light_threshold: 0.3
    nutrient_return_mult: 0.3

  species:
    distance_threshold: 0.5   # genetic distance above which a new species forms

view:
  window_width: 640  # window pixel width
  window_height: 640 # window pixel height
  fps: 30            # frame cap
  title: "Cellular Life: Plants MVP" # window title
```

Notes:
- Larger `world` sizes increase CPU/GPU cost linearly with area.
- Lower `render_every` and higher `fps` make visuals smoother but costlier.
- `max_plants` prevents runaway growth; tune alongside reproduction settings in code if you experiment.
- `initial_plants` controls initial diversity and startup dynamics.
- `reproduction_chance` is stochastic gating for reproduction; lower values slow population growth.
- `max_age` of 0 disables age-based death; higher values allow older plants to persist.
- `seed` ensures identical runs across machines when configs match.

Plants config notes:
- `plants.init` controls the diversity of starting genomes; narrower ranges yield more uniform beginnings.
- `plants.mutation.rate` and sigmas shape evolutionary pace; clips prevent runaway traits.
- `plants.physiology` tunes energy in/out: photosynthesis scaling, nutrient/water usage, shading, and maintenance costs.
- `plants.species.distance_threshold` affects how often new species clusters emerge.

## Project Structure

- `core/ecosystem_sim.py` – main simulation loop and orchestration
- `render/ecosystem_view.py` – performant rendering (biome background, plants, HUD)
- `world/climate.py` – climate & resources (day/night, seasons, rainfall, diffusion)
- `world/grid.py` – occupancy grid
- `plants/plant.py` – plant genome, growth, reproduction, population management
- `utils/rng.py` – random number utility
- `main.py` – CLI entrypoint

## Performance Tips

- Lower window size via `view.window_width`/`window_height`.
- Increase render interval `sim.render_every`.
- Reduce `sim.max_plants`.
- Reduce `world` dimensions in `config/default.yaml`.

## License

MIT
