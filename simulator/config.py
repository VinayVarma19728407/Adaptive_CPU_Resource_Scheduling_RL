
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
