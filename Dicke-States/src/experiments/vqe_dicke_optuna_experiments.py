import sys

sys.path.append("../")

import argparse
import optuna
import pickle
import time

import numpy as np
import pandas as pd

from qiskit.circuit import ClassicalRegister
from qiskit.primitives import BackendEstimatorV2
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from tqdm import tqdm

from utils import ObjectiveOptuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

parser = argparse.ArgumentParser(description="Experiments inputs")
parser.add_argument(
    "--scenario",
    choices=["scenarioI", "scenarioII", "scenarioIII"],
    help="Experiment scenario",
    required=True,
)
parser.add_argument(
    "--device",
    choices=["CPU", "GPU"],
    help="The device that will be used for simulations. CPU or GPU.",
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
    choices=["CMA-ES", "RandomSampler"],
    help="Optimizer name",
    required=True,
)
args = parser.parse_args()

SCENARIO = args.scenario
DEVICE = args.device
METHOD = args.method
try:
    initial_points = np.load(f"{SCENARIO}/initial_points.npy")
    NUM_EXPERIMENTS = int(initial_points.shape[0])
except Exception as _e:
    print(f"{_e}")
    raise
OPTIMIZER = args.optimizer

N_SHOTS_LIST = [2**i for i in range(13)]
MAXITER_LIST = [10**i for i in range(4)]

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

for n_shots in tqdm(N_SHOTS_LIST, desc="num_shots"):
    data = []
    estimator = BackendEstimatorV2(
        backend=simulator, options=dict(default_precision=1 / np.sqrt(n_shots))
    )
    for maxiter in tqdm(MAXITER_LIST, desc="maxiter", leave=False):
        for i in tqdm(range(NUM_EXPERIMENTS), desc="experiments", leave=False):
            objective_optuna = ObjectiveOptuna(
                qc=qc,
                estimator=estimator,
                ising=ising,
                offset=offset,
                bounds=bounds,
            )
            if OPTIMIZER == "CMA-ES":
                x0_dict = {
                    f"x[{j}]": value for j, value in enumerate(initial_points[i])
                }
                sampler = optuna.samplers.CmaEsSampler(
                    x0=x0_dict, use_separable_cma=True
                )
            elif OPTIMIZER == "RandomSampler":
                sampler = optuna.samplers.RandomSampler()
            study = optuna.create_study(sampler=sampler)
            ti = time.time()
            study.optimize(objective_optuna, n_trials=maxiter)
            tf = time.time()

            ansatz = qc.copy()
            params_mapper = {
                param: value
                for param, value in zip(ansatz.parameters, study.best_params.values())
            }
            ansatz = ansatz.assign_parameters(parameters=params_mapper)
            bits = ClassicalRegister(size=len(target), name="bits")
            ansatz.add_register(bits)
            ansatz.measure(
                qubit=ansatz.qregs[0][ising.num_qubits - len(target) :], cbit=bits
            )
            counts = simulator.run(circuits=ansatz, shots=100000).result().get_counts()
            most_probable_bitstring = sorted(
                [(key, value) for key, value in counts.items()],
                key=lambda x: x[1],
                reverse=True,
            )[0][0]
            try:
                target_probability = counts[target] / 1e5
            except:
                target_probability = 0
            data.append(
                [
                    i,
                    OPTIMIZER,
                    n_shots,
                    maxiter,
                    x0_dict,
                    study.best_value,
                    optimal_obj_value - study.best_value,
                    (optimal_obj_value - study.best_value) / optimal_obj_value,
                    most_probable_bitstring,
                    most_probable_bitstring == target,
                    target_probability,
                    study.best_params,
                    tf - ti,
                ]
            )

    df_results = pd.DataFrame(data=data, columns=columns)
    df_results.to_pickle(
        f"{SCENARIO}/results/optuna/{OPTIMIZER}_n_shots_{n_shots}_experiments.pkl"
    )
