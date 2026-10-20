"""Escenarios II y III de Scursulim (febrero de 2026): estructura fiel,
instancia financiera NUEVA si los datos originales no estan disponibles.

Ejecutar desde src/experiments con Pipenv.

MODO DEMOSTRACION (cinco GRUPOS ARTIFICIALES de valores del CSV anonimizado):
  pipenv run python generar_escenarios_multiclase.py --scenario II --demo-groups --n-experiments 100
  pipenv run python generar_escenarios_multiclase.py --scenario III --demo-groups --n-experiments 100

MODO CLASES REALES (archivo con columnas 'asset,asset_class'):
  pipenv run python generar_escenarios_multiclase.py --scenario II \
    --dataset ../datasets/precios_multiclase.csv \
    --classes-csv ../datasets/clases_multiclase.csv --n-experiments 100

ATENCION: genera el circuito de 25 qubits, pero NO ejecuta VQE ni
construye el Statevector completo de 25 qubits (muy costoso en memoria).
"""

import argparse
import hashlib
import itertools
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from docplex.mp.model import Model
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_addon_opt_mapper.converters import OptimizationProblemToQubo
from qiskit_addon_opt_mapper.translators import from_docplex_mp

sys.path.append("../")
from dicke_state_ansatz import DickeStateAnsatz


SCENARIOS = {
    "II": {"weights": [1, 1, 1, 1, 1], "n_params": 20, "n_feasible": 3125},
    "III": {"weights": [2, 2, 1, 1, 3], "n_params": 31, "n_feasible": 25000},
}


def choose_assets(prices, rng, classes_csv, demo_groups):
    """Cinco grupos ordenados de cinco activos cada uno, sin repeticiones."""
    if demo_groups:
        if len(prices.columns) < 25:
            raise ValueError("El CSV requiere al menos 25 activos distintos")
        selected = list(rng.choice(prices.columns.to_numpy(), size=25, replace=False))
        groups = [[str(v) for v in selected[5*i:5*(i+1)]] for i in range(5)]
        labels = [f"grupo_artificial_{i+1}" for i in range(5)]
        return labels, groups

    mapping = pd.read_csv(classes_csv, dtype=str)
    if not {"asset", "asset_class"}.issubset(mapping.columns):
        raise ValueError("--classes-csv debe tener columnas: asset,asset_class")
    mapping = mapping[["asset", "asset_class"]].dropna()
    mapping["asset"] = mapping["asset"].str.strip()
    mapping["asset_class"] = mapping["asset_class"].str.strip()
    if mapping["asset"].duplicated().any():
        raise ValueError("Un activo aparece varias veces en --classes-csv")
    mapping = mapping[mapping["asset"].isin(prices.columns)]
    labels = sorted(mapping["asset_class"].unique().tolist())
    if len(labels) != 5:
        raise ValueError(f"Se requieren EXACTAMENTE cinco clases; detectadas: {labels}")
    groups = []
    for name in labels:
        options = mapping.loc[mapping["asset_class"] == name, "asset"].tolist()
        if len(options) < 5:
            raise ValueError(f"La clase {name!r} solo tiene {len(options)} activos disponibles")
        picked = rng.choice(np.array(options), size=5, replace=False)
        groups.append([str(s) for s in picked])
    return labels, groups


def feasible_bit_matrix(weights):
    """Enumera las combinaciones correctas SIN recorrer los 2**25 estados."""
    combos_per_class = [list(itertools.combinations(range(5), k)) for k in weights]
    n_rows = math.prod(len(c) for c in combos_per_class)
    arr = np.zeros((n_rows, 25), dtype=np.int8)
    for row_idx, class_combos in enumerate(itertools.product(*combos_per_class)):
        for group_idx, local_choices in enumerate(class_combos):
            arr[row_idx, [5*group_idx+j for j in local_choices]] = 1
    return arr


def energy_from_ising(bitstring, terms, offset):
    """Energia del Hamiltoniano diagonal en un bitstring Qiskit q24...q0."""
    result = float(np.real(offset))
    for paulis, coeff in terms:
        parity = 1
        for ch, bit in zip(paulis, bitstring):
            if ch == "Z" and bit == "1":
                parity = -parity
            elif ch not in ("I", "Z"):
                raise ValueError(f"Hamiltoniano no diagonal: {paulis}")
        result += float(np.real(coeff)) * parity
    return result


def build_ansatz(weights, seed):
    qc = QuantumCircuit(25, name="Dicke_multiclase")
    rng = np.random.default_rng(seed)
    local_infeasibility_max = 0.0
    for group_id, k in enumerate(weights):
        block = DickeStateAnsatz().generate_quantum_circuit(
            n=5, k=k, measurement=False, param_name=f"clase{group_id}"
        )
        expected = 5*k - (k*(k+1))//2
        if block.num_qubits != 5 or block.num_parameters != expected:
            raise AssertionError(
                f"Dicke de clase {group_id}: esperado 5 qubits y {expected} parametros; "
                f"observado {block.num_qubits}, {block.num_parameters}"
            )
        # Se verifica cada bloque de 5 qubits: NO usar Statevector del circuito de 25.
        test_params = dict(zip(block.parameters, rng.uniform(0, 2*np.pi, block.num_parameters)))
        small_sv = Statevector.from_instruction(block.assign_parameters(test_params))
        infeasible = sum(
            abs(a)**2 for basis, a in enumerate(small_sv.data)
            if basis.bit_count() != k
        )
        local_infeasibility_max = max(local_infeasibility_max, float(infeasible))
        qc.compose(block, qubits=list(range(group_id*5, (group_id+1)*5)), inplace=True)
    return qc, local_infeasibility_max


def main():
    parser = argparse.ArgumentParser(description="Escenarios multiclase II y III (Dicke)")
    parser.add_argument("--scenario", choices=["II", "III"], required=True)
    parser.add_argument("--dataset", type=Path,
                        default=Path("../datasets/sp500_assets_close_price.csv"),
                        help="CSV de precios: primera columna fecha, las demas columnas activos")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--classes-csv", type=Path,
                      help="CSV con columnas asset,asset_class; cinco clases reales")
    mode.add_argument("--demo-groups", action="store_true",
                      help="Agrupa 25 acciones anonimizadas en cinco grupos ARTIFICIALES")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-experiments", type=int, default=100)
    parser.add_argument("--risk-free", type=float, default=0.0375,
                        help="Provisional: no verificado contra el articulo de febrero")
    parser.add_argument("--q", type=float, default=0.5)
    parser.add_argument("--output", type=Path,
                        help="Carpeta nueva de salida; por defecto scursulim_feb_scenarioII/III")
    args = parser.parse_args()

    if args.n_experiments <= 0:
        parser.error("--n-experiments debe ser positivo")
    if not args.dataset.is_file():
        parser.error(f"No existe el dataset {args.dataset}")
    if args.classes_csv is not None and not args.classes_csv.is_file():
        parser.error(f"No existe --classes-csv {args.classes_csv}")

    cfg = SCENARIOS[args.scenario]
    weights = cfg["weights"]
    output = args.output or Path(f"scursulim_feb_scenario{args.scenario}")
    if output.exists() and any(output.iterdir()):
        parser.error(f"El destino {output} ya contiene archivos; no se sobrescribe")

    prices = pd.read_csv(args.dataset, index_col=0)
    prices.columns = prices.columns.map(str)
    if prices.columns.duplicated().any():
        raise ValueError("Nombres de columnas duplicados en dataset")
    rng = np.random.default_rng(args.seed)
    labels, groups = choose_assets(prices, rng, args.classes_csv, args.demo_groups)
    selected = [asset for group in groups for asset in group]
    if len(selected) != 25 or len(set(selected)) != 25:
        raise AssertionError("Deben seleccionarse 25 activos sin repeticiones")

    close = prices[selected].apply(pd.to_numeric, errors="coerce")
    close = close.replace([np.inf, -np.inf], np.nan)
    if (close <= 0).any().any():
        raise ValueError("Precios no positivos en activos seleccionados")
    returns = close.pct_change(fill_method=None).dropna(how="any")
    if len(returns) < 20:
        raise ValueError(f"Solo {len(returns)} filas de retornos completas")
    mu = returns.mean().to_numpy(dtype=float) * 252.0
    cov = returns.cov().to_numpy(dtype=float) * 252.0
    if not (np.isfinite(mu).all() and np.isfinite(cov).all()):
        raise ValueError("Media/covarianza no finitas")

    portfolios = feasible_bit_matrix(weights)
    if portfolios.shape != (cfg["n_feasible"], 25):
        raise AssertionError(f"Tamaño del espacio factible incorrecto: {portfolios.shape}")
    for group_id, k in enumerate(weights):
        assert np.all(portfolios[:, 5*group_id:5*(group_id+1)].sum(axis=1) == k)
    # Evaluacion vectorizada de la funcion objetivo Markowitz con restriccion exacta.
    B = portfolios.astype(float)
    costs = (args.q * np.einsum("bi,ij,bj->b", B, cov, B, optimize=True)
             - (1-args.q) * (B @ mu) + args.risk_free)
    best_idx = int(np.argmin(costs))
    optimum = float(costs[best_idx])
    target = "".join(map(str, portfolios[best_idx, ::-1]))
    # Lista de carteras con tolerancia absoluta 1e-9.
    n_optima = int(np.sum(np.isclose(costs, optimum, rtol=0, atol=1e-9)))

    # QUBO sin restricciones: la factibilidad se impone en el producto de Dicke.
    model = Model(name=f"scursulim_feb_scenario{args.scenario}")
    x = np.array([model.binary_var(name=f"x({j})") for j in range(25)])
    model.minimize(args.q*(x.T @ cov @ x) - (1-args.q)*(mu @ x) + args.risk_free)
    qp = from_docplex_mp(model=model)
    qubo = OptimizationProblemToQubo(penalty=1e-12).convert(qp)
    ising, offset = qubo.to_ising()
    if ising.num_qubits != 25:
        raise AssertionError(f"Ising tiene {ising.num_qubits} qubits, no 25")

    # Comprobar QUBO<->funcion: muestra mas optimo. No son necesarias 2**25 evaluaciones.
    test_rng = np.random.default_rng(args.seed + 7919)
    sample_ids = test_rng.choice(len(B), min(len(B), 128), replace=False)
    sample_ids = np.unique(np.append(sample_ids, best_idx))
    terms = ising.to_list()
    max_err = 0.0
    for idx in sample_ids:
        bitstring = "".join(map(str, portfolios[idx, ::-1]))
        max_err = max(max_err, abs(energy_from_ising(bitstring, terms, offset) - costs[idx]))
    if max_err > 1e-7:
        raise AssertionError(f"QUBO-Ising inconsistente: error maximo {max_err}")

    circuit, block_bad = build_ansatz(weights, args.seed+123)
    if circuit.num_qubits != 25 or circuit.num_parameters != cfg["n_params"]:
        raise AssertionError(
            f"Ansatz incorrecto: {circuit.num_qubits} qubits, "
            f"{circuit.num_parameters} parametros; esperados 25 y {cfg['n_params']}"
        )
    if block_bad > 1e-10:
        raise AssertionError(f"Dicke local produce estados inviables: {block_bad}")

    init_points = rng.uniform(0, 2*np.pi, (args.n_experiments, circuit.num_parameters))
    metadata = {
        "scenario": args.scenario,
        "note": "REPRODUCCION METODOLOGICA - nueva instancia, NO coincide con la numerica del articulo",
        "groups_are_artificial": bool(args.demo_groups),
        "financial_diversification_is_real": bool(not args.demo_groups),
        "source_dataset": str(args.dataset.resolve()),
        "source_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "class_map_file": str(args.classes_csv.resolve()) if args.classes_csv else None,
        "first_return_index": str(returns.index[0]),
        "last_return_index": str(returns.index[-1]),
        "class_names": labels,
        "assets_by_class": dict(zip(labels, groups)),
        "assets_in_qubit_order_q0_to_q24": selected,
        "qubit_layout": {label: list(range(j*5, (j+1)*5)) for j,label in enumerate(labels)},
        "selected_per_class": weights,
        "n_assets": 25,
        "n_classes": 5,
        "n_selected": int(sum(weights)),
        "n_parameters": circuit.num_parameters,
        "n_feasible": len(portfolios),
        "q": args.q,
        "risk_free": args.risk_free,
        "annualization_mean": 252,
        "annualization_covariance": 252,
        "n_experiments": args.n_experiments,
        "seed": args.seed,
        "optimal_obj_value": optimum,
        "target": target,
        "number_of_tied_optima": n_optima,
        "sampled_ising_error": max_err,
        "max_local_infeasible_p": block_bad,
        "paper_classical_optimum_FOR_REFERENCE_ONLY": (-1.474237 if args.scenario=="II" else -2.00332),
    }
    output.mkdir(parents=True, exist_ok=True)
    for subdir in ["results/scipy", "results/optuna", "results/histories", "figures"]:
        (output / subdir).mkdir(parents=True, exist_ok=True)
    with (output / "scenario_info.pkl").open("wb") as f:
        pickle.dump({
            "optimal_obj_value": optimum,
            "target": target,
            "ising": ising,
            "offset": offset,
            "circuit": circuit,
        }, f)
    np.save(output / "initial_points.npy", init_points)
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nESCENARIO {args.scenario} DE FEBRERO - {'PRUEBA GRUPOS ARTIFICIALES' if args.demo_groups else 'CLASES REALES'}")
    print("Carpeta:", output.resolve())
    print("Periodos de retorno:", returns.index[0], "a", returns.index[-1])
    for j,(lab,group) in enumerate(zip(labels,groups)):
        print(f"  Clase {j+1} ({lab}), seleccionar {weights[j]} de 5: {group}")
    print("Qubits:", circuit.num_qubits)
    print("Parametros:", circuit.num_parameters)
    print("Carteras factibles:", len(portfolios))
    print("Inicializaciones:", init_points.shape)
    print("Optimo clasico de nuestra instancia:", f"{optimum:.9f}")
    print("Bitstring optimo (q24..q0):", target)
    print("Optimos empatados:", n_optima)
    print("Error maximo QUBO/Ising en muestra:", f"{max_err:.3e}")
    print("Probabilidad inviable maxima en bloques:", f"{block_bad:.3e}")
    print("ATENCION: NO EJECUTES TODAVIA VQE de 25 qubits sin comprobar RAM y metodo de simulacion.")
    print("ATENCION: el escenario NO reproduce el valor numerico del paper sin sus datos originales.")


if __name__ == "__main__":
    main()
