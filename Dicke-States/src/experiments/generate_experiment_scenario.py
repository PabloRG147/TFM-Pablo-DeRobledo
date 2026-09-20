import sys

sys.path.append("../")

import argparse
import json
import pickle

import numpy as np
import pandas as pd

from docplex.mp.model import Model
from qiskit.circuit import ParameterVector, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import SparsePauliOp
from qiskit_addon_opt_mapper.converters import OptimizationProblemToQubo
from qiskit_addon_opt_mapper.translators import from_docplex_mp

from dicke_state_ansatz import DickeStateAnsatz

parser = argparse.ArgumentParser(description="Inputs to generate experiments scenarios")
parser.add_argument(
    "--scenario",
    choices=["scenarioI", "scenarioII", "scenarioIII"],
    help="Experiment scenario",
    required=True,
)
parser.add_argument("--seed", type=int, help="The seed number", default=42)
parser.add_argument(
    "n_experiments",
    type=int,
    help="The number of experiments for each number of shots",
    nargs="?",
    default=100,
)
args = parser.parse_args()


SCENARIO = args.scenario
SEED = args.seed
NUM_EXPERIMENTS = args.n_experiments

q = 0.5
return_risk_free = 0.0375
np.random.seed(seed=SEED)
assets_data = pd.read_csv(
    filepath_or_buffer="../datasets/sp500_assets_close_price.csv", index_col=0
)

if SCENARIO == "scenarioI":

    tickers = np.random.choice(assets_data.columns, size=11, replace=False)
    assets_close_price = assets_data[tickers]
    assets_pct_change = assets_close_price.pct_change().dropna()
    covariance_annualized = assets_pct_change.cov() * np.sqrt(252)
    returns_annualized = assets_pct_change.mean() * 252

    k = 4

    model = Model(name="combinatorial_portfolio_optimization")
    x = np.array([model.binary_var(name=f"x({i})") for i in range(len(tickers))])
    model.minimize(
        q * (x.T @ covariance_annualized.values @ x)
        - (1 - q) * (returns_annualized.values @ x)
        + return_risk_free
    )
    model.add_constraint(x.sum() <= k)

    result = model.solve()
    result_array = np.array(
        [
            result.get_value(f"x({i})")
            for i in range(assets_close_price.columns.shape[0])
        ]
    )
    target = result_array.astype(int)
    target = ""
    for item in result_array.astype(int):
        target += str(item)
    target = target[::-1]

    model = Model(name="combinatorial_portfolio_optimization")
    x = np.array([model.binary_var(name=f"x({i})") for i in range(len(tickers))])
    model.minimize(
        q * (x.T @ covariance_annualized.values @ x)
        - (1 - q) * (returns_annualized.values @ x)
        + return_risk_free
    )
    quad_model = from_docplex_mp(model=model)
    qubo_converter = OptimizationProblemToQubo(penalty=1e-12)
    qubo = qubo_converter.convert(quad_model)
    ising, offset = qubo.to_ising()

    identity = SparsePauliOp(data="II", coeffs=[1])
    ising_expanded = ising.tensor(other=identity)

    qubits = QuantumRegister(size=13, name="q")
    params_vec = ParameterVector(name=r"$\theta$", length=2)

    qc = QuantumCircuit(qubits)
    for idx, param in enumerate(params_vec):
        qc.ry(theta=param, qubit=qubits[idx])

    params_names = {1: "alpha", 2: "beta", 3: "gamma", 4: "delta"}
    gate_name = {
        1: r"$\vert D^{11}_{1} \rangle$",
        2: r"$\vert D^{11}_{2} \rangle$",
        3: r"$\vert D^{11}_{3} \rangle$",
        4: r"$\vert D^{11}_{4} \rangle$",
    }

    for k in range(1, 5):
        dicke = DickeStateAnsatz().generate_quantum_circuit(
            n=11, k=k, measurement=False, param_name=params_names[k]
        )
        if k == 1:
            qc.x(qubit=qubits[[i for i in range(2)]])
        elif k == 2:
            qc.x(qubit=qubits[1])
        elif k == 3:
            qc.x(qubit=qubits[0])
        dicke_gate = dicke.to_gate(label=gate_name[k]).control(2)
        qc.append(dicke_gate, qargs=range(qubits.size))
        if k == 1:
            qc.x(qubit=qubits[[i for i in range(2)]])
        elif k == 2:
            qc.x(qubit=qubits[1])
        elif k == 3:
            qc.x(qubit=qubits[0])

    initial_points = (
        2 * np.pi * np.random.random(size=(NUM_EXPERIMENTS, qc.num_parameters))
    )
    np.save(arr=initial_points, file=f"{SCENARIO}/initial_points.npy")

    scenario_info = {
        "optimal_obj_value": result.objective_value,
        "target": target,
        "ising": ising_expanded,
        "offset": offset,
        "circuit": qc,
    }

    with open(file=f"{SCENARIO}/scenario_info.pkl", mode="wb") as f:
        pickle.dump(file=f, obj=scenario_info)
elif SCENARIO == "scenarioII":
    tickers = np.random.choice(assets_data.columns, size=11, replace=False)
    assets_close_price = assets_data[tickers]
    assets_pct_change = assets_close_price.pct_change().dropna()
    covariance_annualized = assets_pct_change.cov() * np.sqrt(252)
    returns_annualized = assets_pct_change.mean() * 252

    k_min = 3
    k_max = 6

    model = Model(name="combinatorial_portfolio_optimization")
    x = np.array([model.binary_var(name=f"x({i})") for i in range(len(tickers))])
    model.minimize(
        q * (x.T @ covariance_annualized.values @ x)
        - (1 - q) * (returns_annualized.values @ x)
        + return_risk_free
    )
    model.add_constraint(x.sum() <= k_max)
    model.add_constraint(x.sum() >= k_min)

    result = model.solve()
    result_array = np.array(
        [
            result.get_value(f"x({i})")
            for i in range(assets_close_price.columns.shape[0])
        ]
    )
    target = result_array.astype(int)
    target = ""
    for item in result_array.astype(int):
        target += str(item)
    target = target[::-1]

    model = Model(name="combinatorial_portfolio_optimization")
    x = np.array([model.binary_var(name=f"x({i})") for i in range(len(tickers))])
    model.minimize(
        q * (x.T @ covariance_annualized.values @ x)
        - (1 - q) * (returns_annualized.values @ x)
        + return_risk_free
    )
    quad_model = from_docplex_mp(model=model)
    qubo_converter = OptimizationProblemToQubo(penalty=1e-12)
    qubo = qubo_converter.convert(quad_model)
    ising, offset = qubo.to_ising()

    identity = SparsePauliOp(data="II", coeffs=[1])
    ising_expanded = ising.tensor(other=identity)

    qubits = QuantumRegister(size=ising_expanded.num_qubits, name="q")
    params_vec = ParameterVector(
        name=r"$\theta$", length=ising_expanded.num_qubits - ising.num_qubits
    )

    qc = QuantumCircuit(qubits)
    for idx, param in enumerate(params_vec):
        qc.ry(theta=param, qubit=qubits[idx])

    params_names = {3: "alpha", 4: "beta", 5: "gamma", 6: "delta"}
    gate_name = {
        3: r"$\vert D^{11}_{3} \rangle$",
        4: r"$\vert D^{11}_{4} \rangle$",
        5: r"$\vert D^{11}_{5} \rangle$",
        6: r"$\vert D^{11}_{6} \rangle$",
    }

    for k in range(k_min, k_max + 1):
        dicke = DickeStateAnsatz().generate_quantum_circuit(
            n=len(tickers), k=k, measurement=False, param_name=params_names[k]
        )
        if k == 3:
            qc.x(
                qubit=qubits[
                    [i for i in range(ising_expanded.num_qubits - ising.num_qubits)]
                ]
            )
        elif k == 4:
            qc.x(qubit=qubits[1])
        elif k == 5:
            qc.x(qubit=qubits[0])
        dicke_gate = dicke.to_gate(label=gate_name[k]).control(2)
        qc.append(dicke_gate, qargs=range(qubits.size))
        if k == 3:
            qc.x(
                qubit=qubits[
                    [i for i in range(ising_expanded.num_qubits - ising.num_qubits)]
                ]
            )
        elif k == 4:
            qc.x(qubit=qubits[1])
        elif k == 5:
            qc.x(qubit=qubits[0])
    initial_points = (
        2 * np.pi * np.random.random(size=(NUM_EXPERIMENTS, qc.num_parameters))
    )
    np.save(arr=initial_points, file=f"{SCENARIO}/initial_points.npy")

    scenario_info = {
        "optimal_obj_value": result.objective_value,
        "target": target,
        "ising": ising_expanded,
        "offset": offset,
        "circuit": qc,
    }

    with open(file=f"{SCENARIO}/scenario_info.pkl", mode="wb") as f:
        pickle.dump(file=f, obj=scenario_info)
elif SCENARIO == "scenarioIII":

    with open(file="../datasets/tickers_by_sectors.json", mode="rt") as f:
        tickers_by_sectors = json.load(f)

    np.random.seed(seed=42)
    assets_classes = np.random.choice(list(tickers_by_sectors.keys()), size=4)
    assets_by_class = {
        value: list(np.random.choice(tickers_by_sectors[value], size=5))
        for value in assets_classes
    }

    constraints_dict = {
        "Energy": ("eq", 3),
        "Financial Services": ("gleq", 1, 2),
        "Real Estate": ("eq", 2),
        "Basic Materials": ("leq", 4),
    }

    tickers = []
    for value in assets_by_class.values():
        tickers.extend(list(value))
    assets_close_price = assets_data[tickers]

    assets_pct_change = assets_close_price.pct_change().dropna()
    covariance_annualized = assets_pct_change.cov() * np.sqrt(252)
    returns_annualized = assets_pct_change.mean() * 252

    constraints_matrix = np.zeros(shape=(len(assets_by_class.keys()), len(tickers)))

    for idx, key in enumerate(assets_by_class.keys()):
        constraints_matrix[idx, :] = assets_close_price.columns.isin(
            assets_by_class[key]
        ).astype(int)

    q = 0.5
    return_risk_free = 0.0375
    model = Model(name="combinatorial_portfolio_optimization")
    x = np.array(
        [
            model.binary_var(name=f"x({i})")
            for i in range(assets_close_price.columns.shape[0])
        ]
    )
    model.minimize(
        q * ((x.T @ covariance_annualized.values @ x))
        - (1 - q) * (returns_annualized.values @ x)
        + return_risk_free
    )
    for idx, key in enumerate(assets_by_class.keys()):
        tp = constraints_dict[key]
        if tp[0] == "eq":
            model.add_constraint(constraints_matrix[idx] @ x == tp[1])
        elif tp[0] == "leq":
            model.add_constraint(constraints_matrix[idx] @ x <= tp[1])
        elif tp[0] == "geq":
            model.add_constraint(constraints_matrix[idx] @ x >= tp[1])
        elif tp[0] == "gleq":
            model.add_constraint(constraints_matrix[idx] @ x >= tp[1])
            model.add_constraint(constraints_matrix[idx] @ x <= tp[2])

    result = model.solve()
    result_array = np.array(
        [
            result.get_value(f"x({i})")
            for i in range(assets_close_price.columns.shape[0])
        ]
    )

    target = result_array.astype(int)
    target = ""
    for item in result_array.astype(int):
        target += str(item)
    target = target[::-1]

    model = Model(name="combinatorial_portfolio_optimization")
    x = np.array([model.binary_var(name=f"x({i})") for i in range(len(tickers))])
    model.minimize(
        q * (x.T @ covariance_annualized.values @ x)
        - (1 - q) * (returns_annualized.values @ x)
        + return_risk_free
    )
    quad_model = from_docplex_mp(model=model)
    qubo_converter = OptimizationProblemToQubo(penalty=1e-12)
    qubo = qubo_converter.convert(quad_model)
    ising, offset = qubo.to_ising()

    identity = SparsePauliOp(data="III", coeffs=[1])
    ising_expanded = ising.tensor(other=identity)

    qubits = QuantumRegister(size=ising_expanded.num_qubits, name="q")
    params_vec = ParameterVector(
        name=r"$\theta$", length=ising_expanded.num_qubits - ising.num_qubits
    )

    qc = QuantumCircuit(qubits)
    for idx, param in enumerate(params_vec):
        qc.ry(theta=param, qubit=qubits[idx])

    qubits_dict = {
        0: [qubits[i] for i in range(3, 8)],
        1: [qubits[i] for i in range(8, 13)],
        2: [qubits[i] for i in range(13, 18)],
        3: [qubits[i] for i in range(18, 23)],
    }
    params_names = {0: "alpha", 1: "beta", 2: "gamma", 3: "delta"}
    gate_name = {
        0: r"$\vert D^{5}_{3} \rangle$",
        1: [r"$\vert D^{5}_{1} \rangle$", r"$\vert D^{5}_{2} \rangle$"],
        2: r"$\vert D^{5}_{2} \rangle$",
        3: [
            r"$\vert D^{5}_{1} \rangle$",
            r"$\vert D^{5}_{2} \rangle$",
            r"$\vert D^{5}_{3} \rangle$",
            r"$\vert D^{5}_{4} \rangle$",
        ],
    }

    for idx, key in enumerate(assets_by_class.keys()):
        tp = constraints_dict[key]
        if tp[0] == "eq":
            dicke = DickeStateAnsatz().generate_quantum_circuit(
                n=len(assets_by_class[key]),
                k=tp[1],
                measurement=False,
                param_name=params_names[idx],
            )
            qc = qc.compose(other=dicke, qubits=qubits_dict[idx])
        elif tp[0] == "leq":
            for k in range(1, tp[1] + 1):
                dicke = DickeStateAnsatz().generate_quantum_circuit(
                    n=len(assets_by_class[key]),
                    k=k,
                    measurement=False,
                    param_name=params_names[idx] + f"_{k}",
                )
                if k == 1:
                    qc.x(qubit=qubits[[i for i in range(2)]])
                elif k == 2:
                    qc.x(qubit=qubits[1])
                elif k == 3:
                    qc.x(qubit=qubits[0])
                dicke_gate = dicke.to_gate(label=gate_name[idx][k - 1]).control(2)
                qc.append(dicke_gate, qargs=[qubits[0], qubits[1]] + qubits_dict[idx])
                if k == 1:
                    qc.x(qubit=qubits[[i for i in range(2)]])
                elif k == 2:
                    qc.x(qubit=qubits[1])
                elif k == 3:
                    qc.x(qubit=qubits[0])
        elif tp[0] == "gleq":
            for k in range(tp[1], tp[2] + 1):
                dicke = DickeStateAnsatz().generate_quantum_circuit(
                    n=len(assets_by_class[key]),
                    k=k,
                    measurement=False,
                    param_name=params_names[idx] + f"_{k}",
                )
                if k == 1:
                    qc.x(qubit=qubits[2])
                dicke_gate = dicke.to_gate(label=gate_name[idx][k - 1]).control(1)
                qc.append(dicke_gate, qargs=[qubits[2]] + qubits_dict[idx])
                if k == 1:
                    qc.x(qubit=qubits[2])

    initial_points = (
        2 * np.pi * np.random.random(size=(NUM_EXPERIMENTS, qc.num_parameters))
    )
    np.save(arr=initial_points, file=f"{SCENARIO}/initial_points.npy")

    scenario_info = {
        "optimal_obj_value": result.objective_value,
        "target": target,
        "ising": ising_expanded,
        "offset": offset,
        "circuit": qc,
    }

    with open(file=f"{SCENARIO}/scenario_info.pkl", mode="wb") as f:
        pickle.dump(file=f, obj=scenario_info)
