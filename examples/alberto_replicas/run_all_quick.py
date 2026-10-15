"""Run all seven Alberto replicas with their short smoke-test settings."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.alberto_replicas.allen_cahn.run import main as allen_cahn
from examples.alberto_replicas.cahn_hilliard.run import main as cahn_hilliard
from examples.alberto_replicas.linear_advection.run import main as linear_advection
from examples.alberto_replicas.shallow_water.run import main as shallow_water
from examples.alberto_replicas.shock_density.run import main as shock_density
from examples.alberto_replicas.shu_osher.run import main as shu_osher
from examples.alberto_replicas.traffic_flow.run import main as traffic_flow


def main():
    runners = (
        ("Allen-Cahn", allen_cahn),
        ("Cahn-Hilliard", cahn_hilliard),
        ("Shallow-Water", shallow_water),
        ("Shock-Density", shock_density),
        ("Shu-Osher", shu_osher),
        ("Linear-Advection", linear_advection),
        ("Traffic-Flow", traffic_flow),
    )
    for name, runner in runners:
        print(f"\n--- {name} ---")
        runner(["--quick"])


if __name__ == "__main__":
    main()
