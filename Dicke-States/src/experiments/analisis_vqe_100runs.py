#!/usr/bin/env python3
"""Análisis genérico de historiales VQE de SciPy y Optuna.

Desde src/experiments (con el mismo Pipenv que los experimentos):
    pipenv run python analizar_vqe.py --scenario scursulim_feb_scenarioI
        --shots 256 4096 --budgets 300
        --optimizers COBYLA CMA-ES --runs 0 1 2 3 4 5 6 7 8 9

Hace 2 tareas independientes, sin reentrenar:
  1. Reevalúa SIN shots el vector de parámetros que obtuvo el menor valor
     de energía OBSERVADA en cada historial, usando Statevector y H + offset.
  2. Dibuja mediana + rango intercuartílico de la mejor energía observada
     durante la optimización (esta curva sigue teniendo ruido de shots).

No cambia historiales, puntos iniciales, escenarios ni scripts de entrenamiento.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
PATRON = re.compile(
    r"^(?P<prefix>.+)_shots(?P<shots>\d+)_budget(?P<budget>\d+)_run(?P<run>\d+)\.pkl$"
)


def nombre_optimizador(prefix):
    for inicio in ("scipy_", "optuna_"):
        if prefix.lower().startswith(inicio):
            return prefix[len(inicio):]
    return prefix


def argumentos():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", required=True, help="Nombre o ruta de la carpeta del escenario")
    p.add_argument("--shots", nargs="+", type=int, help="Shots seleccionados (omitido: todos)")
    p.add_argument("--budgets", nargs="+", type=int, help="Presupuestos seleccionados (omitido: todos)")
    p.add_argument("--optimizers", nargs="+", help="Nombres como COBYLA CMA-ES (omitido: todos)")
    p.add_argument("--runs", nargs="+", type=int, help="Runs seleccionados (omitido: todos)")
    p.add_argument("--output-dir", type=Path, help="Carpeta de salida; defecto: <scenario>/figures/analisis")
    return p.parse_args()


def descubrir(carpeta, args):
    historiales = []
    for ruta in sorted(carpeta.glob("*.pkl")):
        m = PATRON.fullmatch(ruta.name)
        if m is None:
            continue
        info = m.groupdict()
        o = nombre_optimizador(info["prefix"])
        s, b, r = (int(info[k]) for k in ("shots", "budget", "run"))
        if args.shots is not None and s not in args.shots:
            continue
        if args.budgets is not None and b not in args.budgets:
            continue
        if args.runs is not None and r not in args.runs:
            continue
        if args.optimizers is not None and not any(
            elegido.casefold() in (o.casefold(), info["prefix"].casefold())
            for elegido in args.optimizers
        ):
            continue
        historiales.append(dict(path=ruta, optimizer=o, prefix=info["prefix"],
                                shots=s, budget=b, run=r))
    return historiales


def leer_historial(ruta):
    df = pd.read_pickle(ruta)
    if not isinstance(df, pd.DataFrame) or not {"evaluation", "energy", "parameters"}.issubset(df.columns):
        raise ValueError(f"{ruta.name}: se requieren evaluation, energy y parameters")
    df = df.copy()
    df["evaluation"] = pd.to_numeric(df["evaluation"], errors="coerce")
    df["energy"] = pd.to_numeric(df["energy"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["evaluation", "energy"])
    df = df.sort_values("evaluation", kind="stable")
    if df.empty:
        raise ValueError(f"{ruta.name}: no tiene evaluaciones válidas")
    if not (df["evaluation"] >= 1).all() or not np.allclose(df["evaluation"], df["evaluation"].round()):
        raise ValueError(f"{ruta.name}: índices evaluation no enteros positivos")
    df["evaluation"] = df["evaluation"].astype(int)
    # Los índices duplicados son inusuales, pero nos quedamos con el último valor.
    df = df.drop_duplicates(subset="evaluation", keep="last")
    df["best_energy"] = df["energy"].cummin()
    return df


def reevaluar(info, params_guardados, metadata):
    """Se empareja por NOMBRE y no por orden lexicográfico ni identidad del Parameter."""
    from qiskit.quantum_info import Statevector

    qc = info["circuit"]
    ising = info["ising"]
    offset = info["offset"]
    if qc.num_clbits:
        raise ValueError("El circuito contiene bits clásicos; se esperaba el ansatz sin mediciones")
    if qc.num_qubits != ising.num_qubits:
        raise ValueError(f"Qubits del circuito ({qc.num_qubits}) e Ising ({ising.num_qubits}) no coinciden")
    if not isinstance(params_guardados, dict):
        raise ValueError(f"Se esperaba un dict de parámetros, recibido {type(params_guardados).__name__}")

    circuito_por_nombre = {str(p): p for p in qc.parameters}
    guardados_por_nombre = {}
    for clave, valor in params_guardados.items():
        nombre = str(clave)  # Soporta str (SciPy) y Parameter (CMA-ES).
        if nombre in guardados_por_nombre:
            raise ValueError(f"Parámetro duplicado: {nombre}")
        valor = float(np.asarray(valor).item())
        if not np.isfinite(valor):
            raise ValueError(f"Parámetro no finito: {nombre}")
        guardados_por_nombre[nombre] = valor

    faltan = sorted(set(circuito_por_nombre) - set(guardados_por_nombre))
    sobran = sorted(set(guardados_por_nombre) - set(circuito_por_nombre))
    if faltan or sobran:
        raise ValueError(f"Nombres de parámetros incompatibles. Faltan: {faltan[:5]}; sobran: {sobran[:5]}")

    # Qiskit determina su propio orden de parámetros; el mapa por nombre evita
    # asignaciones erróneas como confundir alpha[2] con alpha[20].
    asignaciones = {param: guardados_por_nombre[nombre]
                    for nombre, param in circuito_por_nombre.items()}
    preparado = qc.assign_parameters(asignaciones, inplace=False)
    state = Statevector.from_instruction(preparado)
    exp_h = complex(state.expectation_value(ising))
    valor = exp_h + complex(np.asarray(offset).item())
    if abs(valor.imag) > 1e-7:
        raise ValueError(f"La energía tiene parte imaginaria inesperada: {valor}")
    energia_exacta = float(valor.real)

    probs = np.abs(np.asarray(state.data)) ** 2
    if not np.isclose(probs.sum(), 1.0, atol=1e-9):
        raise ValueError("La distribución de Statevector no está normalizada")

    objetivo = info.get("target")
    prob_optimo = float("nan")
    if isinstance(objetivo, str) and len(objetivo) == qc.num_qubits and set(objetivo) <= {"0", "1"}:
        prob_optimo = float(probs[int(objetivo, 2)])

    fuga = float("nan")
    if "n_selected" in metadata:
        k = int(metadata["n_selected"])
        fuga = float(sum(float(p) for i, p in enumerate(probs) if i.bit_count() != k))
        if fuga > 1e-7:
            raise ValueError(f"El ansatz genera probabilidad {fuga:.3g} en estados inviables")
    return energia_exacta, prob_optimo, fuga


def guardar_grafica(grupo, curvas, optimo, carpeta, sufijo):
    shots, budget = grupo
    fig, ax = plt.subplots(figsize=(10, 6))
    eje = np.arange(1, budget + 1)
    for optimizador in sorted(curvas):
        vectores = np.asarray(curvas[optimizador], dtype=float)
        mediana = np.nanmedian(vectores, axis=0)
        q1 = np.nanpercentile(vectores, 25, axis=0)
        q3 = np.nanpercentile(vectores, 75, axis=0)
        (linea,) = ax.plot(eje, mediana, lw=2, label=f"{optimizador} (n={len(vectores)})")
        ax.fill_between(eje, q1, q3, color=linea.get_color(), alpha=0.20)

    ax.axhline(optimo, linestyle="--", color="black", lw=1.5,
               label=f"Óptimo clásico = {optimo:.6f}")
    ax.set_xlabel("Evaluaciones de la función objetivo")
    ax.set_ylabel("Mejor energía observada (mediana e IQR)")
    ax.set_title(f"Convergencia agregada | {shots} shots | presupuesto {budget}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    salida = carpeta / f"convergencia_agregada_shots{shots}_budget{budget}_{sufijo}.png"
    fig.savefig(salida, dpi=300)
    plt.close(fig)
    return salida


def grafica_boxplot(df, carpeta, sufijo):
    """Comparación exacta por grupo, para separar el ranking del ruido de shots."""
    combos = sorted(set(zip(df["shots"], df["optimizer"])))
    valores = [df[(df.shots == s) & (df.optimizer == o)]["exact_energy"].to_numpy()
               for s, o in combos]
    etiquetas = [f"{o}\n{s} shots" for s, o in combos]
    fig, ax = plt.subplots(figsize=(max(8, 1.7 * len(combos)), 6))
    ax.boxplot(valores, showmeans=True)
    ax.set_xticks(np.arange(1, len(combos) + 1), etiquetas)
    ax.set_ylabel("Energía exacta del mejor punto observado")
    ax.set_title("Reevaluación sin ruido de shots")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    ruta = carpeta / f"energia_exacta_boxplot_{sufijo}.png"
    fig.savefig(ruta, dpi=300)
    plt.close(fig)
    return ruta


def sufijo_archivos(items):
    shots = "-".join(map(str, sorted(set(h["shots"] for h in items))))
    budgets = "-".join(map(str, sorted(set(h["budget"] for h in items))))
    indices = sorted(set(h["run"] for h in items))
    digest = hashlib.sha256(",".join(map(str, indices)).encode()).hexdigest()[:8]
    runs = f"{indices[0]}-{indices[-1]}_n{len(indices)}_h{digest}"
    opts = "-".join(sorted(set(h["optimizer"] for h in items)))
    # Evita que líneas de comando distintas se pisen al cambiar los runs.
    return re.sub(r"[^a-zA-Z0-9_.-]", "-", f"shots-{shots}_budget-{budgets}_opt-{opts}_runs-{runs}")


def main():
    args = argumentos()
    scenario = Path(args.scenario).expanduser()
    if not scenario.is_absolute():
        scenario = BASE / scenario
    scenario = scenario.resolve()
    history_dir = scenario / "results" / "histories"
    if not history_dir.is_dir():
        raise FileNotFoundError(f"No encuentro {history_dir}")

    info_path = scenario / "scenario_info.pkl"
    if not info_path.is_file():
        raise FileNotFoundError(f"No encuentro {info_path}")
    info = pd.read_pickle(info_path)
    optimo = float(info["optimal_obj_value"])
    metadata_path = scenario / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}

    items = descubrir(history_dir, args)
    if not items:
        raise ValueError("No hay historiales con estos filtros. Comprueba --scenario/--shots/--budgets/--runs")
    grupos_esperados = {(h["shots"], h["budget"], h["optimizer"], h["run"]) for h in items}
    if len(grupos_esperados) != len(items):
        raise ValueError("Existen nombres de historiales duplicados para una misma configuración")

    # Informar ausencias explícitas para evitar comparaciones incompletas silenciosas.
    shot_set = sorted(set(args.shots if args.shots is not None else [h["shots"] for h in items]))
    budget_set = sorted(set(args.budgets if args.budgets is not None else [h["budget"] for h in items]))
    run_set = sorted(set(args.runs if args.runs is not None else [h["run"] for h in items]))
    opt_set = sorted(set(args.optimizers if args.optimizers is not None else [h["optimizer"] for h in items]))
    ausentes = [f"{s} shots / budget {b} / {o} / run {r}"
               for s in shot_set for b in budget_set for o in opt_set for r in run_set
               if (s, b, nombre_optimizador(o), r) not in grupos_esperados]
    if ausentes:
        print(f"AVISO: faltan {len(ausentes)} historiales de las combinaciones solicitadas.")
        for falta in ausentes[:12]:
            print("  -", falta)
        if len(ausentes) > 12:
            print("  ...")
        print("El resumen mostrará cuántas ejecuciones contiene cada grupo.")

    outdir = args.output_dir if args.output_dir is not None else (scenario / "figures" / "analisis")
    if not outdir.is_absolute():
        outdir = BASE / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    sufijo = sufijo_archivos(items)
    filas = []
    curvas_grupos = {}

    for idx, h in enumerate(items, start=1):
        hist = leer_historial(h["path"])
        mejor_fila = hist.loc[hist["energy"].idxmin()]
        exacta, p_objetivo, fuga = reevaluar(info, mejor_fila["parameters"], metadata)
        gap = exacta - optimo
        if gap < -1e-6:
            raise ValueError(
                f"{h['path'].name}: energía exacta {exacta:.8f} es inferior al óptimo clásico "
                f"{optimo:.8f}. Revisa Hamiltoniano/ansatz/asignación de parámetros."
            )
        filas.append({
            "shots": h["shots"], "budget": h["budget"],
            "optimizer": h["optimizer"], "run": h["run"],
            "evaluations": len(hist),
            "best_evaluation": int(mejor_fila["evaluation"]),
            "observed_best_energy": float(mejor_fila["energy"]),
            "exact_energy": exacta,
            "exact_gap": gap,
            "observed_minus_exact": float(mejor_fila["energy"]) - exacta,
            "probability_optimal_bitstring": p_objetivo,
            "infeasible_probability": fuga,
            "filename": h["path"].name,
        })
        # Rellenamos las evaluaciones posteriores a una parada temprana con el
        # mejor valor alcanzado; NO se inventan nuevas evaluaciones.
        curva = hist.set_index("evaluation")["best_energy"]
        curva = curva.reindex(range(1, h["budget"] + 1)).ffill().to_numpy(dtype=float)
        curvas_grupos.setdefault((h["shots"], h["budget"]), {}).setdefault(
            h["optimizer"], []).append(curva)
        print(f"[{idx}/{len(items)}] {h['optimizer']} shots={h['shots']} "
              f"budget={h['budget']} run={h['run']}: "
              f"observada={float(mejor_fila['energy']):.6f}; exacta={exacta:.6f}")

    resultados = pd.DataFrame(filas)
    resumen = (resultados.groupby(["shots", "budget", "optimizer"], as_index=False)
               .agg(runs=("run", "count"),
                    mean_exact=("exact_energy", "mean"),
                    median_exact=("exact_energy", "median"),
                    std_exact=("exact_energy", "std"),
                    best_exact=("exact_energy", "min"),
                    worst_exact=("exact_energy", "max"),
                    mean_gap=("exact_gap", "mean"),
                    mean_p_optimum=("probability_optimal_bitstring", "mean"),
                    mean_evaluations=("evaluations", "mean")))

    file_results = outdir / f"reevaluaciones_{sufijo}.csv"
    file_summary = outdir / f"estadisticas_exactas_{sufijo}.csv"
    resultados.to_csv(file_results, index=False)
    resumen.to_csv(file_summary, index=False)
    print("\nÓPTIMO CLÁSICO:", f"{optimo:.8f}")
    print("\nESTADÍSTICAS DE ENERGÍA EXACTA (sin ruido de shots):")
    print(resumen.round(6).to_string(index=False))

    print("\nCOMPARACIONES EMPAREJADAS (misma inicialización):")
    for (s, b), subset in resultados.groupby(["shots", "budget"]):
        paired = subset.pivot_table(index="run", columns="optimizer", values="exact_energy")
        if {"COBYLA", "CMA-ES"}.issubset(paired.columns):
            paired = paired.dropna(subset=["COBYLA", "CMA-ES"])
            c = int((paired["COBYLA"] < paired["CMA-ES"]).sum())
            d = int((paired["CMA-ES"] < paired["COBYLA"]).sum())
            print(f"{s} shots, budget {b}: COBYLA {c}/{len(paired)}; CMA-ES {d}/{len(paired)}")

    print("\nARCHIVOS GENERADOS:")
    print(file_results)
    print(file_summary)
    for grupo, curvas in sorted(curvas_grupos.items()):
        print(guardar_grafica(grupo, curvas, optimo, outdir, sufijo))
    print(grafica_boxplot(resultados, outdir, sufijo))
    print("\nNOTA: las curvas agregadas representan mínimos OBSERVADOS (con ruido). ")
    print("La reevaluación exacta corresponde SOLO al mejor punto observado de cada historial.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
