"""Generador independiente de escenario I (reproduccion metodologica de Scursulim, febrero 2026).

No reproduce necesariamente la instancia numerica del articulo: usa el CSV anonimizado
que acompana al repositorio posterior y decisiones de preprocesado explicitadas aqui.

Ejecutar desde src/experiments:
  pipenv run python generate_scursulim_feb_scenarioI.py --n-experiments 2

No modifica scenarioI/ ni los resultados previos.
"""

import argparse
import hashlib
import itertools
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from docplex.mp.model import Model
from qiskit_addon_opt_mapper.converters import OptimizationProblemToQubo
from qiskit_addon_opt_mapper.translators import from_docplex_mp

import sys
sys.path.append("../")
from dicke_state_ansatz import DickeStateAnsatz


def energy_for_bits(bitstring, pauli_terms, offset):
    """Valor de H + offset sobre un estado computacional (Hamiltoniano diagonal)."""
    total = float(np.real(offset))
    for label, coefficient in pauli_terms:
        sign = 1
        for bit, pauli in zip(bitstring, label):
            if pauli == "Z" and bit == "1":
                sign = -sign
            elif pauli not in ("Z", "I"):
                raise ValueError("El Hamiltoniano no es diagonal en la base Z")
        total += float(np.real(coefficient)) * sign
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-experiments", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=Path, default=Path("../datasets/sp500_assets_close_price.csv"))
    parser.add_argument("--output", type=Path, default=Path("scursulim_feb_scenarioI"))
    parser.add_argument("--risk-free", type=float, default=0.0375,
                        help="Valor heredado del repositorio posterior, no verificado para febrero")
    args = parser.parse_args()
    if args.n_experiments < 1:
        parser.error("--n-experiments debe ser >= 1")
    if (args.output / "scenario_info.pkl").exists() or (args.output / "initial_points.npy").exists():
        parser.error(f"{args.output}: los datos ya existen; elige otra carpeta con --output")

    n, k, q = 10, 4, 0.5
    if not args.dataset.exists():
        raise FileNotFoundError(args.dataset)

    raw = pd.read_csv(args.dataset, index_col=0)
    if raw.shape[1] < n:
        raise ValueError(f"El CSV tiene {raw.shape[1]} columnas, se necesitan al menos {n}")
    rng = np.random.default_rng(args.seed)
    tickers = [str(x) for x in rng.choice(raw.columns.to_numpy(), size=n, replace=False)]
    close = raw[tickers].apply(pd.to_numeric, errors="coerce")
    returns = close.pct_change(fill_method=None).dropna(how="any")
    if len(returns) < 3:
        raise ValueError("No hay suficientes retornos validos en el CSV seleccionado")

    # Convencion financiera explicita (NO confirmada como la empleada en febrero):
    # anualizacion habitual de la media y de la matriz de covarianzas.
    mu = returns.mean().to_numpy(dtype=float) * 252.0
    cov = returns.cov().to_numpy(dtype=float) * 252.0
    if not np.isfinite(mu).all() or not np.isfinite(cov).all():
        raise ValueError("Se obtuvieron medias o covarianzas no finitas")

    def objective(bits):
        vec = np.asarray(bits, dtype=float)
        return float(q * (vec @ cov @ vec) - (1.0 - q) * (mu @ vec) + args.risk_free)

    # Para n=10 y k=4 enumerar las 210 carteras factibles es exacto y economico.
    feasible = []
    for indexes in itertools.combinations(range(n), k):
        x = np.zeros(n, dtype=int)
        x[list(indexes)] = 1
        feasible.append((objective(x), x))
    optimum, best_x = min(feasible, key=lambda elem: elem[0])
    target = "".join(str(b) for b in best_x[::-1])  # orden Qiskit q9...q0

    # Restriccion de cardinalidad incorporada SOLO en el ansatz; sin penalizacion QUBO.
    model = Model(name="scursulim_feb_scenarioI")
    x = np.array([model.binary_var(name=f"x({i})") for i in range(n)])
    model.minimize(q * (x.T @ cov @ x) - (1.0 - q) * (mu @ x) + args.risk_free)
    qp = from_docplex_mp(model=model)
    qubo = OptimizationProblemToQubo(penalty=1e-12).convert(qp)
    ising, offset = qubo.to_ising()

    circuit = DickeStateAnsatz().generate_quantum_circuit(
        n=n, k=k, measurement=False, param_name="alpha"
    )
    if circuit.num_qubits != 10 or circuit.num_parameters != 30:
        raise AssertionError("El ansatz no tiene 10 qubits y 30 parametros")

    # Comprobacion integral QUBO->Ising: cada una de las 210 carteras coincide
    # con la funcion financiera, teniendo en cuenta el offset.
    pauli_terms = ising.to_list()
    max_error = 0.0
    for value, solution in feasible:
        bits = "".join(str(b) for b in solution[::-1])
        error = abs(energy_for_bits(bits, pauli_terms, offset) - value)
        max_error = max(max_error, error)
    if max_error > 1e-7:
        raise AssertionError(f"Inconsistencia entre funcion objetivo e Ising: {max_error}")

    initial_points = rng.uniform(0, 2 * np.pi, size=(args.n_experiments, circuit.num_parameters))
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ["results/scipy", "results/optuna", "results/histories", "figures"]:
        (args.output / name).mkdir(parents=True, exist_ok=True)
    np.save(args.output / "initial_points.npy", initial_points)
    with (args.output / "scenario_info.pkl").open("wb") as file:
        pickle.dump({
            "optimal_obj_value": optimum,
            "target": target,
            "ising": ising,
            "offset": offset,
            "circuit": circuit,
        }, file)

    metadata = {
        "note": "Reproduccion metodologica, no numerica exacta, del escenario I (febrero de 2026)",
        "source_dataset": str(args.dataset.resolve()),
        "source_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "first_index": str(returns.index[0]),
        "last_index": str(returns.index[-1]),
        "assets": tickers,
        "n_assets": n,
        "n_selected": k,
        "q": q,
        "risk_free": args.risk_free,
        "annualization_mean": 252,
        "annualization_covariance": 252,
        "seed": args.seed,
        "n_experiments": args.n_experiments,
        "optimal_obj_value": optimum,
        "target": target,
        "max_ising_cost_error": max_error,
    }
    with (args.output / "metadata.json").open("w") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print("\nESCENARIO I DE FEBRERO (reproduccion metodologica)")
    print("Carpeta:", args.output.resolve())
    print("Filas de retornos:", len(returns))
    print("Periodo disponible:", returns.index[0], "a", returns.index[-1])
    print("Activos:", tickers)
    print("Qubits:", circuit.num_qubits)
    print("Parametros:", circuit.num_parameters)
    print("Carteras factibles:", len(feasible))
    print("Puntos iniciales:", initial_points.shape)
    print("Optimo clasico de NUESTRA instancia:", f"{optimum:.6f}")
    print("Bitstring optimo:", target)
    print("Error maximo Ising vs. objetivo:", f"{max_error:.3e}")
    print("Aviso: NO comparar el valor numerico anterior directamente con -0.818106")


if __name__ == "__main__":
    main()
