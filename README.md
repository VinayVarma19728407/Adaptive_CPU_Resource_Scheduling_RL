@'
# Adaptive Multi-Core CPU Scheduler using Reinforcement Learning

## Overview

This project investigates adaptive multi-core CPU scheduling using workload
analysis, data preprocessing, scenario generation, and a custom scheduling
simulator.

The current simulator is a prototype. Gymnasium and PPO integration should
be treated as future work until implemented and tested.

## Project Structure

- `data_cleaning/` — Scripts for cleaning source datasets.
- `data_documentation/` — Dataset documentation utilities.
- `data_preprocessing/` — Dataset preprocessing scripts.
- `dataset` — Contains all the dataset excluding the raw, cleaned, preprocessed
- `simulator/` — Scheduling simulator, event queue, machine state, and task loading.
- `experiments/` — Experiment scenarios, inputs, and results.


## Requirements

Python 3.13 is the current development environment.

Install dependencies:

```bash
python -m pip install -r requirements.txt