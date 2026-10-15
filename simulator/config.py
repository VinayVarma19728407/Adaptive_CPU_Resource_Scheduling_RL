"""
============================================================
File: config.py
Module: simulator
============================================================
Central configuration and path registry for the RL-based
CPU scheduling simulator. All project-wide constants, dataset
paths, and tunable simulator settings are defined here so that
every other module in the simulator package can import from a
single authoritative source.

Key Components:
  - PROJECT_ROOT: Resolved absolute path to the repository root,
    used to build all other paths relative to the project.
  - DATASET_DIR / PREPROCESSED_DIR / CLEANED_DIR: Directory
    constants pointing to the preprocessed and cleaned dataset
    sub-directories under `dataset/`.
  - BATCH_TASK_FILE, MACHINE_META_FILE: Paths to the primary
    preprocessed data files consumed by task_data.py.
  - TRAINING_FILE, VALIDATION_FILE, TESTING_FILE: Paths to the
    train/val/test splits of the batch instance dataset.
  - MACHINE_USAGE_FILE: Path to the cleaned machine-usage CSV.
  - CSV_CHUNK_SIZE: Row batch size for memory-efficient chunked
    CSV reading (250,000 rows per chunk).
  - RANDOM_SEED: Global reproducibility seed (42).
  - MAX_CANDIDATE_MACHINES: Maximum number of machines presented
    to the RL agent as action candidates per scheduling step.
  - REQUIRE_CPU_DEMAND, REQUIRE_MEMORY_DEMAND,
    REQUIRE_BASELINE_DURATION: Boolean guards that control
    whether tasks missing these fields are rejected during
    scenario validation.
  - ELIGIBLE_MACHINE_STATUS: Integer status code (1 = USING)
    that a machine must have to be admitted to the simulator pool.
  - validate_required_files(): Raises FileNotFoundError if any
    mandatory dataset file is absent from disk.

Inputs:
  - No runtime inputs; values are derived from __file__ at import
    time and from hard-coded constants.

Outputs:
  - Module-level constants and the validate_required_files()
    helper consumed by task_data.py, machine_state.py,
    scenario_loader.py, and other simulator sub-modules.
============================================================
"""

from pathlib import Path

# Project root is the parent of the simulator directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
PREPROCESSED_DIR = DATASET_DIR / "preprocessed"
CLEANED_DIR = DATASET_DIR / "cleaned"

BATCH_TASK_FILE = PREPROCESSED_DIR / "batch_task.csv"
MACHINE_META_FILE = PREPROCESSED_DIR / "machine_meta.csv"

TRAINING_FILE = PREPROCESSED_DIR / "batch_instance_training.csv"
VALIDATION_FILE = PREPROCESSED_DIR / "batch_instance_validation.csv"
TESTING_FILE = PREPROCESSED_DIR / "batch_instance_testing.csv"

MACHINE_USAGE_FILE = CLEANED_DIR / "machine_usage.csv"

# Memory-conscious data access.
CSV_CHUNK_SIZE = 250_000
RANDOM_SEED = 42

# Initial simulator configuration.
MAX_CANDIDATE_MACHINES = 20

# A task scenario must have reliable demand and duration information.
REQUIRE_CPU_DEMAND = True
REQUIRE_MEMORY_DEMAND = True
REQUIRE_BASELINE_DURATION = True

# Initial eligible machine status from the validated data contract.
ELIGIBLE_MACHINE_STATUS = 1

# These are paths and design settings, not claims that every file exists.
def validate_required_files():
    required = {
        "batch_task": BATCH_TASK_FILE,
        "machine_meta": MACHINE_META_FILE,
    }

    missing = {
        name: path
        for name, path in required.items()
        if not path.is_file()
    }

    if missing:
        details = "\n".join(
            f"{name}: {path}" for name, path in missing.items()
        )
        raise FileNotFoundError(
            "Required simulator data files are missing:\n" + details
        )
