@'
# Adaptive Multi-Core CPU Scheduler using Reinforcement Learning

## Overview

This project investigates adaptive multi-core CPU scheduling using workload
analysis, data preprocessing, scenario generation, and a custom scheduling
simulator.

The current simulator is a prototype. Gymnasium and PPO integration should
be treated as future work until implemented and tested.

## Project Structure

- `analysis/` — Workload, telemetry, and resource analysis.
- `data_cleaning/` — Scripts for cleaning source datasets.
- `data_documentation/` — Dataset documentation utilities.
- `data_preprocessing/` — Dataset preprocessing scripts.
- `environment/` — Environment-related configuration and data structures.
- `simulator/` — Scheduling simulator, event queue, machine state, and task loading.
- `scenario/` — Workload scenario generation and validation.
- `validation/` — Data quality, relationship, and temporal validation.
- `experiments/` — Experiment scenarios, inputs, and results.
- `tests/` — Simulator and validation tests.
- `docs/` — Technical documentation.
- `Old/` — Previous implementation, excluded from Git.

## Requirements

Python 3.13 is the current development environment.

Install dependencies:

```bash
python -m pip install -r requirements.txt