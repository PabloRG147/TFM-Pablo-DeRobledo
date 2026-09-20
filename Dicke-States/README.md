# Quantum Optimization with Pure and Mixed Dicke States

[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Description

This repository contains the code and data for the research paper *"Pure and mixed Dicke state ansatz for equality and inequality constraints in variational quantum eigensolver"*. We propose an ansatz based on pure or mixed Dicke states for the Variational Quantum Eigensolver (VQE) to handle combinatorial optimization problems with equality and inequality constraints on the Hamming weight of the decision variables, eliminating the need for penalty terms in the objective function. The framework is validated in the context of combinatorial portfolio optimization across three experimental scenarios with increasing feasible search space sizes.

## Repository Structure

- src/
    - datasets/              # Preprocessed asset data (anonymized)
    - dicke_state_ansatz/    # Dicke state ansatz implementation
    - experiments/           # Experiment setup and execution scripts
    - hardware_calibration_data/ # IBM Quantum device calibration data
    - notebooks/             # Jupyter notebooks for analysis and visualization
    - scripts/               # Data download and preprocessing scripts
    - utils/                 # Auxiliary and utility functions
- Pipfile                    # Python dependencies and version
- README.md
- .gitignore
- LICENSE

## How to Install

### Requirements
- Python 3.11
- For GPU acceleration: CUDA 11

### Steps

1. Clone this repository:
```bash
git clone https://github.com/jvscursulim/quantum-optimization-dicke-states.git
cd quantum-optimization-dicke-states
```

2. Create and activate a Python virtual environment:
```bash
python -m venv env
source env/bin/activate
```

3. Update `pip` and install `pipenv`:
```bash
pip install --upgrade pip
pip install pipenv
```

4. Install all project dependencies:
```bash
pipenv install
```

<details>
<summary>Python libraries</summary>

- numpy
- matplotlib
- seaborn
- pandas
- scipy
- pylatexenc
- qiskit
- qiskit-ibm-runtime
- qiskit-aer-gpu-cu11
- ipykernel
- plotly
- qiskit-addon-opt-mapper
- cplex
- gurobipy
- cmaes
- optuna
- yfinance
- tqdm
- nbformat
- qiskit-experiments
</details>

## How to Use

### 1. Prepare the data

Run the script `prepare_data.sh` to download and preprocess asset data from Yahoo Finance:
```bash
cd src/scripts
bash prepare_data.sh
```
> **Note:** A preprocessed CSV file with anonymized asset identifiers is already available in the `data/` folder for reproducibility.

### 2. Generate experiment setup

Prepare the Ising model, ansatz, initial points, optimal objective value, and target decision variables for each scenario:
```bash
cd src/experiments
python generate_experiment_scenario.py n_experiments --scenario scenario_name --seed seed_number
```
> **Note:** All scenarios in this study were generated with `n_experiments=100` and `seed=42`.

### 3. Run the experiments

#### Using Scipy
```bash
python vqe_dicke_scipy_experiments.py --optimizer optimizer_name --scenario scenario_name --device device_name --method method_name
```

Available optimizers: `["Nelder-Mead", "Powell", "CG", "BFGS", "Newton-CG", "L-BFGS-B", "TNC", "COBYLA", "COBYQA", "SLSQP", "trust-constr", "dogleg", "trust-ncg", "trust-exact", "trust-krylov"]`

#### Using Optuna
```bash
python vqe_dicke_optuna_experiments.py --optimizer optimizer_name --scenario scenario_name --device device_name --method method_name
```

Available optimizers: `["CMA-ES", "RandomSampler"]`

#### Common options (Scipy and Optuna)
| Argument | Options |
|----------|---------|
| `--scenario` | `scenarioI`, `scenarioII`, `scenarioIII` |
| `--device` | `CPU`, `GPU` |
| `--method` | `statevector`, `matrix_product_state`, `tensor_network` |

> For additional information about simulation methods please consult the [Qiskit Aer Documentation](https://qiskit.github.io/qiskit-aer/).

### 4. Generate the efficient frontier
```bash
python generate_scenario_efficient_frontier.py --scenario scenario_name --seed seed_number
```
> **Note:** Use `seed=42` to ensure consistency with the experimental results reported in the paper.

## Disclaimer

This repository is purely academic. The portfolio optimization problem is used solely as a benchmark for combinatorial optimization methods and does not constitute financial advice or investment recommendations.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{scursulim2026mixture,
  title={Pure and mixed Dicke state ansatz for equality and inequality constraints in variational quantum eigensolver},
  author={Scursulim, J. V. S.},
  journal={},
  year={2026}
}
```

## References

All references consulted during this research can be found in the manuscript: [Pure and mixed Dicke state ansatz for equality and inequality constraints in variational quantum eigensolver](https://arxiv.org/abs/2606.08504)
