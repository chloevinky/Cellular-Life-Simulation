from __future__ import annotations

import argparse
from pathlib import Path

from core.ecosystem_sim import run_with_config_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Miniature Earth - Ecosystem Simulation")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML or project root")
    args = parser.parse_args()

    cfg_path: str | Path | None = args.config
    run_with_config_path(cfg_path)


if __name__ == "__main__":
    main()
