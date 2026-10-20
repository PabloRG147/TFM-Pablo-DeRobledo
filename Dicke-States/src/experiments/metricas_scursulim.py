#!/usr/bin/env python3
"""Metricas comparables con Fig. 5 y Tabla 3 de Scursulim (febrero 2026).

No ejecuta optimizadores. Reconstruye el estado exacto del mejor punto
OBSERVADO de cada historial y comprueba el optimo entre las carteras factibles.

Ejemplo desde src/experiments:
  pipenv run python metricas_scursulim.py \
    --scenario scursulim_feb_scenarioI_controlado \
    --shots 4096 --budget 1000 --optimizers COBYLA CMA-ES \
    --runs $(seq 0 99)

Atencion: la frecuencia modal es IDEAL (Statevector), no la frecuencia
estimada con las 4096 medidas concretas de cada ejecucion del paper.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Reutilizamos las funciones que ya se validaron sobre el escenario de 100 runs.
from validacion_estadistica_vqe_100runs import energia_diagonal, preparar_estado

BASE = Path(__file__).resolve().parent
PREFIJOS = {"COBYLA": "scipy_COBYLA", "CMA-ES": "CMA-ES"}


def args_cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--optimizers", nargs="+", choices=sorted(PREFIJOS), default=["COBYLA", "CMA-ES"])
    parser.add_argument("--runs", nargs="+", type=int, default=list(range(100)))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.shots <= 0 or args.budget <= 0 or not args.runs or min(args.runs) < 0 or len(set(args.runs)) != len(args.runs):
        parser.error("shots y budget deben ser positivos; runs distintos y no negativos")
    return args


def energy_probs(qc, bindings, energies, feasible_mask):
    state, _ = preparar_estado(qc, bindings)
    prob = np.abs(np.asarray(state.data)) ** 2
    if not np.isclose(prob.sum(), 1.0, atol=1e-8):
        raise ValueError("El estado no esta normalizado")
    leak = float(prob[~feasible_mask].sum())
    if leak > 1e-7:
        raise ValueError(f"El ansatz genero probabilidad inviable: {leak}")
    return float(np.dot(prob, energies)), prob, leak


def main():
    a = args_cli()
    scenario = Path(a.scenario).expanduser()
    if not scenario.is_absolute():
        scenario = BASE / scenario
    scenario = scenario.resolve()
    info_file = scenario / "scenario_info.pkl"
    points_file = scenario / "initial_points.npy"
    if not info_file.is_file() or not points_file.is_file():
        raise FileNotFoundError("Falta scenario_info.pkl o initial_points.npy en " + str(scenario))
    info = pd.read_pickle(info_file)
    meta_file = scenario / "metadata.json"
    metadata = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
    qc = info["circuit"]
    initial_points = np.load(points_file)
    if max(a.runs) >= len(initial_points) or initial_points.shape[1] != qc.num_parameters:
        raise ValueError("initial_points.npy no cubre los runs solicitados o tiene numero de parametros incorrecto")

    n = qc.num_qubits
    target = info["target"]
    if len(target) != n or set(target) - {"0", "1"}:
        raise ValueError("Bitstring target invalido")
    k = int(metadata.get("n_selected", target.count("1")))
    if target.count("1") != k:
        raise ValueError("El target no cumple la cardinalidad")
    spectrum = energia_diagonal(info["ising"], info["offset"], n)
    indices = np.arange(1 << n)
    feasible_mask = np.array([int(idx).bit_count() == k for idx in indices], dtype=bool)
    feasible = indices[feasible_mask]
    emin_feasible = float(np.min(spectrum[feasible_mask]))
    e0 = float(info["optimal_obj_value"])
    imax_global = float(np.max(spectrum))
    imax_feasible = float(np.max(spectrum[feasible_mask]))
    emin_global = float(np.min(spectrum))
    target_index = int(target, 2)
    if abs(spectrum[target_index] - emin_feasible) > 1e-8 or abs(e0 - emin_feasible) > 1e-8:
        raise ValueError("El target y/o el optimo guardado no representan el minimo factible")
    if not imax_global > e0 or not imax_feasible > e0:
        raise ValueError("Emax debe ser mayor que E0 para definir el ratio")
    print("\nREFERENCIAS ENERGETICAS")
    print(f"  Optimo factible E0: {e0:.9f} | Emax global H: {imax_global:.9f}")
    print(f"  Emax factible: {imax_feasible:.9f} | minimo global H: {emin_global:.9f}")
    if emin_global < e0 - 1e-8:
        print("  AVISO: el minimo global sin restricciones esta por debajo del optimo factible.")
        print("  Se utiliza E0 FACTIBLE para evaluar los resultados de cartera.")
    print("  ar_paper usa Emax GLOBAL del Hamiltoniano; ar_factible usa Emax de carteras factibles.")

    sorted_feasible = feasible[np.argsort(spectrum[feasible])]
    top5 = sorted_feasible[:5]
    initial_cache = {}
    for run in a.runs:
        values = initial_points[run]
        bindings = {p: float(x) for p, x in zip(qc.parameters, values)}
        energy, _, _ = energy_probs(qc, bindings, spectrum, feasible_mask)
        initial_cache[run] = energy

    all_records = []
    for opt in a.optimizers:
        for run in a.runs:
            filename = f"{PREFIJOS[opt]}_shots{a.shots}_budget{a.budget}_run{run}.pkl"
            filename_path = scenario / "results" / "histories" / filename
            if not filename_path.is_file():
                raise FileNotFoundError(filename_path)
            hist = pd.read_pickle(filename_path)
            if not {"energy", "evaluation", "parameters"}.issubset(hist.columns):
                raise ValueError(f"Faltan columnas en {filename}")
            hist = hist.copy()
            hist["energy"] = pd.to_numeric(hist["energy"], errors="coerce")
            hist["evaluation"] = pd.to_numeric(hist["evaluation"], errors="coerce")
            hist = hist.replace([np.inf, -np.inf], np.nan).dropna(subset=["energy", "evaluation"])
            if hist.empty:
                raise ValueError(f"Historial vacio: {filename}")
            best = hist.loc[hist["energy"].idxmin()]
            energy, probabilities, leak = energy_probs(qc, best["parameters"], spectrum, feasible_mask)
            p_target = float(probabilities[target_index])
            max_p = float(np.max(probabilities))
            co_modes = int(np.count_nonzero(np.abs(probabilities - max_p) < 1e-12))
            success = bool(p_target >= max_p - 1e-12 and co_modes == 1)
            p0 = initial_cache[run]
            row = {
                "optimizer": opt, "run": run, "shots": a.shots, "budget": a.budget,
                "evaluations": len(hist), "best_evaluation": int(best["evaluation"]),
                "initial_energy_exact": p0, "final_energy_exact": energy,
                "best_observed_energy": float(best["energy"]),
                "ar_paper_initial": (imax_global - p0)/(imax_global - e0),
                "ar_paper_final": (imax_global - energy)/(imax_global - e0),
                "ar_feasible_initial": (imax_feasible - p0)/(imax_feasible - e0),
                "ar_feasible_final": (imax_feasible - energy)/(imax_feasible - e0),
                "p_target": p_target, "is_target_unique_mode": int(success),
                "p_target_ge_095": int(p_target >= 0.95),
                "success_with_p_ge_095": int(success and p_target >= 0.95),
                "max_probability": max_p, "mode_ties": co_modes,
                "most_probable_bitstring": format(int(np.argmax(probabilities)), f"0{n}b"),
                "p_infeasible": leak,
            }
            for rank, idx in enumerate(top5, start=1):
                row[f"p_top_{rank}"] = float(probabilities[idx])
            row["p_other_than_top5"] = float(1 - np.sum(probabilities[top5]))
            all_records.append(row)
        print(f"  Analizadas {len(a.runs)} ejecuciones de {opt}")
    df = pd.DataFrame(all_records)
    summary = df.groupby("optimizer", sort=False).agg(
        n=("run", "size"), mean_ar_initial=("ar_paper_initial", "mean"),
        mean_ar_final=("ar_paper_final", "mean"),
        std_ar_final=("ar_paper_final", "std"),
        median_ar_final=("ar_paper_final", "median"),
        mean_p_target=("p_target", "mean"),
        target_mode_count=("is_target_unique_mode", "sum"),
        p_ge_095_count=("p_target_ge_095", "sum"),
        mode_and_p_ge_095=("success_with_p_ge_095", "sum"),
        mean_evaluations=("evaluations", "mean"),
    ).reset_index()
    print("\nMETRICAS COMPARABLES CON ECUACION 7 / TABLA 3 (Statevector IDEAL)")
    print(summary.round(6).to_string(index=False))
    top_summary = df.groupby("optimizer", sort=False)[[f"p_top_{i}" for i in range(1, 6)] + ["p_other_than_top5"]].mean().reset_index()
    print("\nPROBABILIDAD MEDIA DE LAS 5 MEJORES CARTERAS FACTIBLES")
    print("  Bitstrings: " + ", ".join(f"{format(int(idx), f'0{n}b')}" for idx in top5))
    print(top_summary.round(6).to_string(index=False))

    out = a.output_dir or (scenario / "figures" / "metricas_scursulim")
    if not out.is_absolute():
        out = BASE / out
    out.mkdir(parents=True, exist_ok=True)
    runs_sorted = sorted(a.runs)
    token = f"shots{a.shots}_budget{a.budget}_runs{runs_sorted[0]}-{runs_sorted[-1]}_n{len(a.runs)}"
    path_detail = out / f"detalle_{token}.csv"
    path_summary = out / f"resumen_{token}.csv"
    path_top = out / f"top5_{token}.csv"
    df.to_csv(path_detail, index=False)
    summary.to_csv(path_summary, index=False)
    top_summary.to_csv(path_top, index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5.5))
    plot_data, ticks, xpos = [], [], []
    for i, opt in enumerate(a.optimizers):
        block = df.loc[df.optimizer == opt]
        plot_data.extend([block["ar_paper_initial"], block["ar_paper_final"]])
        xpos.extend([i*3+1, i*3+2])
        ticks.extend([f"{opt}\ninicial", f"{opt}\nfinal"])
    ax.boxplot(plot_data, positions=xpos, widths=0.65, showmeans=True)
    ax.set_xticks(xpos, ticks)
    ax.axhline(0.9, linestyle="--", linewidth=1, label="Ratio = 0.9 (Fig. 5)")
    ax.axhline(1, linestyle=":", linewidth=1, label="Optimo = 1")
    ax.set_ylabel("Ratio de aproximacion exacto (Emax global)")
    ax.set_title(f"Scursulim - antes/despues | {a.shots} shots | budget {a.budget}")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    path_fig = out / f"aproximacion_antes_despues_{token}.png"
    fig.savefig(path_fig, dpi=220)
    plt.close(fig)
    print("\nARCHIVOS:")
    for path in [path_detail, path_summary, path_top, path_fig]:
        print(" ", path)
    print("\nIMPORTANTE: p_target, frecuencias y ratios se han calculado sobre estados exactos")
    print("del mejor punto OBSERVADO durante el entrenamiento, no sobre medidas")
    print("empiricas tomadas en el paper ni necesariamente sobre el ultimo punto del optimizador.")


if __name__ == "__main__":
    main()
