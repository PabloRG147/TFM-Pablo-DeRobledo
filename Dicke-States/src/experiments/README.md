# Experiments

## Description

This folder contains Python scripts for experiment preparation and execution, including
Ising model formulation, ansatz setup, initial point generation, and results collection.

## Experimental Scenarios

### Scenario I

Scenario I consists of a portfolio optimization with 11 assets subject to an inequality
constraint $k \leq 4$, where $k$ denotes the number of selected assets (Hamming weight
of the decision variables). The feasible search space has size $|\mathcal{F}| = 561$.
This scenario is designed to validate the mixed Dicke state ansatz for encoding a
single inequality constraint by construction, eliminating the need for penalty terms
in the objective function.

### Scenario II

Scenario II extends the previous one by introducing two simultaneous inequality
constraints, $k \geq 3$ and $k \leq 6$, establishing both a lower and an upper bound
on the number of selected assets. The feasible search space has size $|\mathcal{F}| = 1419$.
This scenario tests the ansatz in settings where the feasible region is defined by
a floor and a ceiling on the Hamming weight.

### Scenario III

Scenario III is designed to validate the tensor product construction of the ansatz
for problems combining equality and inequality constraints simultaneously. Assets are
grouped into 4 sectors of the S&P 500 index (Energy, Financial Services, Real Estate,
and Basic Materials), with 5 assets per sector and independent constraints per sector.
The feasible search space has size $|\mathcal{F}| = 45000$, making this the most
challenging scenario and the one where the advantage over random search is most
pronounced.

## How to Use

### 1. Prepare experiment setup

Generate the Ising model, ansatz, initial points, optimal objective value, and target
decision variables for each scenario:
```bash
python generate_experiment_scenario.py n_experiments --scenario scenario_name --seed seed_number
```
> **Note:** All scenarios in this study were prepared with `n_experiments=100` and `seed=42`.

### 2. Run the experiments

#### Using Scipy
```bash
python vqe_dicke_scipy_experiments.py --optimizer optimizer_name --scenario scenario_name --device device_name --method method_name
```

Available optimizers: `["Nelder-Mead", "Powell", "CG", "BFGS", "Newton-CG", "L-BFGS-B",
"TNC", "COBYLA", "COBYQA", "SLSQP", "trust-constr", "dogleg", "trust-ncg", "trust-exact",
"trust-krylov"]`

#### Using Optuna
```bash
python vqe_dicke_optuna_experiments.py --optimizer optimizer_name --scenario scenario_name --device device_name --method method_name
```

Available optimizers: `["CMA-ES", "RandomSampler"]`

#### Common options

| Argument | Options |
|----------|---------|
| `--scenario` | `scenarioI`, `scenarioII`, `scenarioIII` |
| `--device` | `CPU`, `GPU` |
| `--method` | `statevector`, `matrix_product_state`, `tensor_network` |

> For additional information about simulation methods please consult the
[Qiskit Aer Documentation](https://qiskit.github.io/qiskit-aer/).

### 3. Generate the efficient frontier
```bash
python generate_scenario_efficient_frontier.py --scenario scenario_name --seed seed_number
```
> **Note:** Use `seed=42` to ensure consistency with the experimental results
reported in the paper.