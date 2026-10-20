import sys

sys.path.append("../")

import argparse
import pickle
import scipy
import time

import numpy as np
import pandas as pd

from qiskit.circuit import ClassicalRegister
from qiskit.primitives import BackendEstimatorV2
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from tqdm import tqdm
from pathlib import Path
from seeded_estimator import ReseedingEstimator, seed_for

from utils import objective_scipy

parser = argparse.ArgumentParser(description="Experiments inputs")
parser.add_argument(
    "--scenario",
    choices=["scenarioI", "scenarioII", "scenarioIII", "scursulim_feb_scenarioI", "scursulim_feb_scenarioI_controlado", "scursulim_feb_scenarioII_demo",
"scursulim_feb_scenarioIII_demo", "scursulim_feb_scenarioI_real", "scursulim_feb_scenarioII_real", "scursulim_feb_scenarioIII_real"],
    help="Experiment scenario",
    required=True,
)
parser.add_argument(
    "--device",
    choices=["CPU", "GPU"],
    help="The device that will be used for simulations.",
    default="CPU",
)
parser.add_argument(
    "--method",
    choices=["statevector", "matrix_product_state", "tensor_network"],
    help="The method that will be used for simulations.",
    default="statevector",
)
parser.add_argument(
    "--optimizer",
    choices=[
        "Nelder-Mead",
        "Powell",
        "CG",
        "BFGS",
        "Newton-CG",
        "L-BFGS-B",
        "TNC",
        "COBYLA",
        "COBYQA",
        "SLSQP",
        "trust-constr",
        "dogleg",
        "trust-ncg",
        "trust-exact",
        "trust-krylov"
    ],
    help="Optimizer name",
    required=True,
)
parser.add_argument("--shots", type=int, nargs="+", default=[256],
                    help="Número(s) de shots; ejemplo: --shots 256 4096")
parser.add_argument("--budgets", type=int, nargs="+", default=[100, 300],
                    help="Presupuestos maxiter/n_trials; ejemplo: --budgets 100 300")
parser.add_argument("--runs", type=int, nargs="+", default=[0, 1],
                    help="Índices de puntos iniciales; ejemplo: --runs 0 1")
parser.add_argument("--overwrite", action="store_true",
                    help="Permitir sobrescribir historiales/resultados existentes")
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Semilla base para las simulaciones"
)

args = parser.parse_args()
SCENARIO = args.scenario
DEVICE = args.device
METHOD = args.method
try:
    initial_points = np.load(f"{SCENARIO}/initial_points.npy")
    NUM_EXPERIMENTS = len(initial_points)
except Exception as _e:
    print(f"{_e}")
    raise
OPTIMIZER = args.optimizer

### N_SHOTS_LIST se define mediante --shots ###
N_SHOTS_LIST = list(dict.fromkeys(args.shots))
MAXITER_LIST = list(dict.fromkeys(args.budgets))
RUNS = list(dict.fromkeys(args.runs))
if any(s <= 0 for s in N_SHOTS_LIST):
    parser.error("--shots requiere enteros positivos")
if any(b <= 0 for b in MAXITER_LIST):
    parser.error("--budgets requiere enteros positivos")
if not RUNS or any(r < 0 or r >= len(initial_points) for r in RUNS):
    parser.error(
        f"--runs debe contener índices entre 0 y {len(initial_points)-1}; "
        "regenera initial_points.npy con más puntos si los necesitas"
    )

def _etiqueta_indices(valores):
    if len(valores) > 12:
        import hashlib
        sufijo = hashlib.sha256(",".join(map(str, valores)).encode()).hexdigest()[:8]
        return f"{min(valores)}-{max(valores)}-{sufijo}"
    return "-".join(map(str, valores))

RESULTS_TAG = f"budgets_{_etiqueta_indices(MAXITER_LIST)}_runs_{_etiqueta_indices(RUNS)}"
_results_dir = Path(SCENARIO) / "results" / "scipy"
_histories_dir = Path(SCENARIO) / "results" / "histories"
_results_dir.mkdir(parents=True, exist_ok=True)
_histories_dir.mkdir(parents=True, exist_ok=True)

# Comprobar ANTES de gastar tiempo en optimizaciones.
if not args.overwrite:
    conflictos = []
    for _shots in N_SHOTS_LIST:
        _result_file = _results_dir / f"{OPTIMIZER}_n_shots_{_shots}_{RESULTS_TAG}_experiments.pkl"
        if _result_file.exists():
            conflictos.append(_result_file)
        for _budget in MAXITER_LIST:
            for _run in RUNS:
                _history_file = _histories_dir / (
                    f"scipy_{OPTIMIZER}_shots{_shots}_budget{_budget}_run{_run}.pkl"
                )
                if _history_file.exists():
                    conflictos.append(_history_file)
    if conflictos:
        parser.error(
            "Ya existen resultados/historiales de estas configuraciones. "
            "No se sobrescriben automáticamente. Para repetirlas intencionadamente, "
            "añade --overwrite. Primer conflicto: " + str(conflictos[0])
        )

print(f"Configuración CLI: shots={N_SHOTS_LIST}, budgets={MAXITER_LIST}, runs={RUNS}")

simulator = AerSimulator(method=METHOD, device=DEVICE)
pm = generate_preset_pass_manager(backend=simulator, optimization_level=0)

with open(file=f"{SCENARIO}/scenario_info.pkl", mode="rb") as f:
    scenario_info = pickle.load(f)

optimal_obj_value = scenario_info["optimal_obj_value"]
target = scenario_info["target"]
ising = scenario_info["ising"]
offset = scenario_info["offset"]
qc = scenario_info["circuit"]
qc = pm.run(qc)

bounds = [(0, 2 * np.pi) for _ in range(qc.num_parameters)]

columns = [
    "experiment_index",
    "optimizer",
    "n_shots",
    "n_iter",
    "budget",
    "initial_point",
    "obj_value",
    "delta_obj_value",
    "percentual_error",
    "most_probable_bitstring",
    "is_target_most_probable",
    "target_probability_after_training",
    "parameters_after_training",
    "time_spent",
]

print(f"NUM_EXPERIMENTS real: {NUM_EXPERIMENTS}")
print(f"Presupuestos: {MAXITER_LIST}")

for n_shots in tqdm(N_SHOTS_LIST, desc="num_shots"):
    data = []
    for maxiter in tqdm(MAXITER_LIST, desc="maxiter", leave=False):
        for i in tqdm(RUNS, desc="experiments", leave=False):
        
            seed_train = seed_for(
                args.seed, n_shots, maxiter, i, stream=0
            )

            seed_measure = seed_for(
                args.seed, n_shots, maxiter, i, stream=1
            )

            estimator = ReseedingEstimator(
                backend=simulator,
                shots=n_shots,
                seed=seed_train
            )

            history = []

            print(
                f"Run={i}, shots={n_shots}, budget={maxiter}, "
                f"seed_train={seed_train}"
            )
            
            ti = time.time()
            result_scipy = scipy.optimize.minimize(
                fun=objective_scipy,
                x0=initial_points[i],
                method=OPTIMIZER,
                bounds=bounds,
                args=(estimator, qc, ising, offset, history),
                options=dict(maxiter=maxiter),
            )
            tf = time.time()
            
            history_dir = Path(SCENARIO) / "results" / "histories"
            history_dir.mkdir(parents=True, exist_ok=True)
            
            history_df = pd.DataFrame([
                {
                    "evaluation": j,
                    "energy": float(np.asarray(energy).item()),
                    "parameters": {str(k): float(v) for k, v in params.items()
                    },
                }
                for j, (energy, params) in enumerate(history, start=1)
            ])

            history_df.to_pickle(history_dir/ f"scipy_{OPTIMIZER}_shots{n_shots}_budget{maxiter}_run{i}.pkl")
            ansatz = qc.copy()
            params_mapper = {
                param: value for param, value in zip(ansatz.parameters, result_scipy.x)
            }
            ansatz = ansatz.assign_parameters(parameters=params_mapper)
            bits = ClassicalRegister(size=len(target), name="bits")
            ansatz.add_register(bits)
            ansatz.measure(qubit=ansatz.qregs[0][ising.num_qubits-len(target):], cbit=bits)
            counts = simulator.run(circuits=ansatz, shots=4096, seed_simulator=seed_measure).result().get_counts()
            most_probable_bitstring = sorted(
                [(key, value) for key, value in counts.items()],
                key=lambda x: x[1],
                reverse=True,
            )[0][0]
            try:
                target_probability = counts[target] / 4096
            except:
                target_probability = 0
            data.append(
                [
                    i,
                    OPTIMIZER,
                    n_shots,
                    result_scipy.nfev,
                    maxiter,
                    initial_points[i],
                    result_scipy.fun,
                    optimal_obj_value - result_scipy.fun,
                    (optimal_obj_value - result_scipy.fun) / optimal_obj_value,
                    most_probable_bitstring,
                    most_probable_bitstring == target,
                    target_probability,
                    result_scipy.x,
                    tf - ti,
                ]
            )
            
    df_results = pd.DataFrame(data=data, columns=columns)
    print("Número de resultados acumulados:", len(data))
    print("Índices de ejecución:", df_results["experiment_index"].tolist())
    df_results.to_pickle(f"{SCENARIO}/results/scipy/{OPTIMIZER}_n_shots_{n_shots}_{RESULTS_TAG}_experiments.pkl")
