import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from simulator.environment import SchedulingEnvironment

ROOT = Path(__file__).resolve().parents[1]
scenario = ROOT / "experiments" / "scenarios" / "heavy_scenario_seed42.csv"
output_dir = ROOT / "experiments" / "results"
output_dir.mkdir(parents=True, exist_ok=True)

for policy in ["first_fit", "least_loaded"]:
    env = SchedulingEnvironment(scenario, placement_policy=policy)
    outcome = env.run()

    result_path = output_dir / f"heavy_{policy}_seed42_results.csv"
    outcome["results"].to_csv(result_path, index=False)

    print(f"\nPOLICY: {policy}")
    for key, value in outcome["metrics"].items():
        print(f"{key}: {value}")
    print(f"Machines used: {outcome['results']['machine_id'].nunique()}")
    print(f"Results saved: {result_path}")
