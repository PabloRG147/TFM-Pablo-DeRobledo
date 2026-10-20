#!/usr/bin/env python3
"""Comparación de COBYLA y CMA-ES a igual número efectivo de evaluaciones.

Para cada run: si COBYLA termina tras N evaluaciones, localiza el mejor
punto OBSERVADO en las primeras N evaluaciones de CMA-ES. Reevalúa SIN
ruido de shots ambos puntos usando el mismo Hamiltoniano y Statevector.
No reentrena ni modifica los historiales. Requiere el módulo de análisis
anterior (analisis_vqe_100runs.py o analisis_vqe.py) en la misma carpeta.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest

try:
    from analisis_vqe_100runs import leer_historial, reevaluar
except ImportError:
    from analisis_vqe import leer_historial, reevaluar


def bootstrap_interval(values, n_samples=20000, seed=42):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_samples, len(values)))
    replicas = values[indices].mean(axis=1)
    return np.percentile(replicas, [2.5, 97.5])


def best_row(hist):
    return hist.loc[hist["energy"].idxmin()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", required=True)
    p.add_argument("--shots", type=int, default=4096)
    p.add_argument("--budget", type=int, default=1000)
    p.add_argument("--runs", nargs="+", type=int, default=list(range(100)))
    args = p.parse_args()

    base = Path(__file__).resolve().parent
    scenario = Path(args.scenario).expanduser()
    if not scenario.is_absolute():
        scenario = base / scenario
    scenario = scenario.resolve()
    carpeta = scenario / "results" / "histories"
    if not carpeta.is_dir():
        p.error(f"No existe: {carpeta}")
    info = pd.read_pickle(scenario / "scenario_info.pkl")
    metadata_file = scenario / "metadata.json"
    metadata = json.loads(metadata_file.read_text(encoding="utf-8")) if metadata_file.exists() else {}
    optimo = float(info["optimal_obj_value"])

    filas = []
    for run in args.runs:
        c_path = carpeta / f"scipy_COBYLA_shots{args.shots}_budget{args.budget}_run{run}.pkl"
        m_path = carpeta / f"CMA-ES_shots{args.shots}_budget{args.budget}_run{run}.pkl"
        if not c_path.exists() or not m_path.exists():
            raise FileNotFoundError(f"Run {run}: falta {c_path.name if not c_path.exists() else m_path.name}")
        cob = leer_historial(c_path)
        cma = leer_historial(m_path)
        n = len(cob)
        if not np.array_equal(cob["evaluation"].to_numpy(), np.arange(1, n + 1)):
            raise ValueError(f"COBYLA run {run}: índices de evaluación no consecutivos")
        if len(cma) < n or not np.array_equal(cma["evaluation"].iloc[:n].to_numpy(), np.arange(1, n + 1)):
            raise ValueError(f"CMA-ES run {run}: faltan las primeras {n} evaluaciones")

        cb = best_row(cob)
        mb = best_row(cma.iloc[:n])
        mf = best_row(cma)
        ec, pc, _ = reevaluar(info, cb["parameters"], metadata)
        em, pm, _ = reevaluar(info, mb["parameters"], metadata)
        ef, pf, _ = reevaluar(info, mf["parameters"], metadata)
        if min(ec, em, ef) < optimo - 1e-6:
            raise ValueError(f"Run {run}: energía exacta inferior al óptimo clásico; revisar parámetros")
        filas.append({
            "run": run, "evaluaciones_iguales": n,
            "eval_cobyla_mejor": int(cb["evaluation"]),
            "eval_cma_mejor_prefijo": int(mb["evaluation"]),
            "energia_cobyla_exacta": ec,
            "energia_cma_mismo_coste_exacta": em,
            "energia_cma_completa_exacta": ef,
            "p_target_cobyla": pc,
            "p_target_cma_mismo_coste": pm,
            "p_target_cma_completa": pf,
            "delta_cobyla_menos_cma_exacta": ec - em,
            "gap_cobyla": ec - optimo,
            "gap_cma_mismo_coste": em - optimo,
        })

    df = pd.DataFrame(filas).sort_values("run")
    d = df["delta_cobyla_menos_cma_exacta"].to_numpy()
    n_cma = int((d > 1e-12).sum())
    n_cob = int((d < -1e-12).sum())
    n_ties = len(d) - n_cma - n_cob
    p_value = binomtest(min(n_cma, n_cob), n=n_cma + n_cob, p=0.5).pvalue if n_cma + n_cob else 1.0
    ci = bootstrap_interval(d)

    print("\nCOMPARACIÓN A IGUAL COSTE (ENERGÍAS REEVALUADAS SIN SHOTS)")
    print(f"Escenario: {scenario.name} | shots={args.shots} | budget={args.budget} | n={len(df)}")
    print(f"Óptimo clásico: {optimo:.9f}")
    print(f"Evaluaciones medias de COBYLA (= límite del prefijo de CMA-ES): {df['evaluaciones_iguales'].mean():.2f}")
    for col in ("energia_cobyla_exacta", "energia_cma_mismo_coste_exacta", "energia_cma_completa_exacta"):
        print(f"{col}: media={df[col].mean():.6f}, mediana={df[col].median():.6f}, desviación={df[col].std(ddof=1):.6f}")
    print(f"Diferencia media emparejada COBYLA - CMA-ES a igual coste: {d.mean():+.6f}")
    print(f"IC bootstrap 95 % para la diferencia media: [{ci[0]:+.6f}, {ci[1]:+.6f}]")
    print(f"Victorias CMA-ES={n_cma}, COBYLA={n_cob}, empates={n_ties}")
    print(f"Test de signos bilateral: p={p_value:.6g}")
    print("\nNota: se selecciona cada punto por su menor energía OBSERVADA dentro del prefijo, y se reevalúa su energía EXACTA.")
    print("La comparación es retrospectiva sobre trayectorias iniciadas con budget 1000; no equivale a reentrenar CMA-ES con budget N.")
    output = scenario / "figures" / "comparacion_coste_igual"
    output.mkdir(parents=True, exist_ok=True)
    filename = f"coste_igual_shots{args.shots}_budget{args.budget}_runs{min(args.runs)}-{max(args.runs)}_n{len(args.runs)}.csv"
    df.to_csv(output / filename, index=False)
    print(f"\nCSV guardado: {output / filename}")


if __name__ == "__main__":
    main()
