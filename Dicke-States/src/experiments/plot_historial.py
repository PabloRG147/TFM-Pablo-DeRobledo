"""
Graficas de convergencia VQE a partir de historiales de SciPy y Optuna.

Ejemplo:
    pipenv run python plot_historial.py --scenario scursulim_feb_scenarioI --shots 256 --budgets 100 300 --optimizers COBYLA CMA-ES --runs 0 1

Los historiales deben llamarse: <prefijo>_shots<N>_budget<B>_run<R>.pkl
Donde, por ejemplo, <prefijo> es scipy_COBYLA o CMA-ES.
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(__file__).resolve().parent
PATRON = re.compile(
    r"^(?P<prefix>.+)_shots(?P<shots>\d+)_budget(?P<budget>\d+)_run(?P<run>\d+)\.pkl$"
)


def nombre_optimizador(prefix):
    """Nombre visual, manteniendo también el prefijo original para el archivo."""
    for source_prefix in ("scipy_", "optuna_"):
        if prefix.lower().startswith(source_prefix):
            return prefix[len(source_prefix):]
    return prefix


def buscar_historiales(carpeta):
    """Devuelve metadatos de historiales sin abrir sus archivos."""
    archivos = []
    for ruta in sorted(carpeta.glob("*.pkl")):
        match = PATRON.fullmatch(ruta.name)
        if match is None:
            continue
        d = match.groupdict()
        archivos.append({
            "path": ruta,
            "prefix": d["prefix"],
            "optimizer": nombre_optimizador(d["prefix"]),
            "shots": int(d["shots"]),
            "budget": int(d["budget"]),
            "run": int(d["run"]),
        })
    return archivos


def opciones():
    parser = argparse.ArgumentParser(
        description="Representa curvas VQE desde historiales de SciPy/Optuna. "
                    "Los filtros omitidos incluyen todos los valores disponibles."
    )
    parser.add_argument("--scenario", required=True,
                        help="Carpeta del escenario (relativa a src/experiments, o absoluta).")
    parser.add_argument("--shots", type=int, nargs="+", metavar="N",
                        help="Uno o varios valores de shots. Si se omite, todos.")
    parser.add_argument("--budgets", type=int, nargs="+", metavar="N",
                        help="Uno o varios presupuestos. Si se omite, todos.")
    parser.add_argument("--optimizers", nargs="+", metavar="NOMBRE",
                        help="Optimizadores (p. ej., COBYLA CMA-ES); acepta también el prefijo exacto.")
    parser.add_argument("--runs", type=int, nargs="+", metavar="N",
                        help="Índices de ejecución (p. ej., 0 1). Si se omite, todos.")
    parser.add_argument("--metric", choices=("best", "energy"), default="best",
                        help="best: mínimo acumulado (defecto); energy: valor observado.")
    parser.add_argument("--output-dir", type=Path,
                        help="Carpeta de salida (por defecto: <scenario>/figures).")
    parser.add_argument("--no-optimum", action="store_true",
                        help="No mostrar la línea horizontal del óptimo clásico.")
    parser.add_argument("--list", action="store_true", dest="list_only",
                        help="Listar los historiales disponibles, sin generar gráficas.")
    return parser.parse_args()


def etiqueta_archivo(s):
    """Sanitiza solo el nombre del archivo de salida."""
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", s).strip("-") or "all"


def cargar_optimo(scenario, no_optimum):
    if no_optimum:
        return None
    ruta = scenario / "scenario_info.pkl"
    if not ruta.exists():
        print(f"Aviso: no existe {ruta}; se omitirá la referencia clásica.")
        return None
    try:
        info = pd.read_pickle(ruta)
        return float(info["optimal_obj_value"])
    except (KeyError, ValueError, TypeError, OSError) as exc:
        print(f"Aviso: no se puede leer el óptimo clásico ({exc}).")
        return None


def cargar_historial(ruta):
    try:
        df = pd.read_pickle(ruta)
    except Exception as exc:
        print(f"Aviso: no se ha podido abrir {ruta.name}: {exc}")
        return None
    if not isinstance(df, pd.DataFrame) or not {"evaluation", "energy"}.issubset(df.columns):
        print(f"Aviso: formato no válido en {ruta.name} (faltan evaluation/energy).")
        return None
    df = df.copy()
    df["evaluation"] = pd.to_numeric(df["evaluation"], errors="coerce")
    df["energy"] = pd.to_numeric(df["energy"], errors="coerce")
    df = df.dropna(subset=["evaluation", "energy"]).sort_values("evaluation")
    if df.empty:
        print(f"Aviso: historial vacío o sin evaluaciones válidas: {ruta.name}")
        return None
    df["best_energy"] = df["energy"].cummin()
    return df


def main():
    args = opciones()
    scenario = Path(args.scenario).expanduser()
    if not scenario.is_absolute():
        scenario = BASE / scenario
    scenario = scenario.resolve()
    carpeta = scenario / "results" / "histories"
    if not carpeta.is_dir():
        print(f"Error: no existe la carpeta de historiales: {carpeta}", file=sys.stderr)
        return 1

    disponibles = buscar_historiales(carpeta)
    if not disponibles:
        print(f"Error: no se encontraron historiales *_shotsN_budgetB_runR.pkl en {carpeta}",
              file=sys.stderr)
        return 1

    if args.list_only:
        print(f"Historiales disponibles en {carpeta}:")
        for h in disponibles:
            print(f"  {h['optimizer']:16} shots={h['shots']:<6} "
                  f"budget={h['budget']:<6} run={h['run']:<3} archivo={h['path'].name}")
        return 0

    selected = disponibles
    if args.shots is not None:
        selected = [h for h in selected if h["shots"] in args.shots]
    if args.budgets is not None:
        selected = [h for h in selected if h["budget"] in args.budgets]
    if args.runs is not None:
        selected = [h for h in selected if h["run"] in args.runs]
    if args.optimizers is not None:
        wanted = {x.casefold() for x in args.optimizers}
        selected = [h for h in selected if h["optimizer"].casefold() in wanted
                    or h["prefix"].casefold() in wanted]

    if not selected:
        print("No hay historiales que cumplan los filtros. "
              "Usa --scenario ... --list para inspeccionar los disponibles.", file=sys.stderr)
        return 1

    # Avisos específicos si el usuario solicitó explícitamente combinaciones ausentes.
    shots = sorted(set(args.shots if args.shots is not None else (h["shots"] for h in selected)))
    budgets = sorted(set(args.budgets if args.budgets is not None else (h["budget"] for h in selected)))
    runs = sorted(set(args.runs if args.runs is not None else (h["run"] for h in selected)))
    prefixes = sorted({h["prefix"] for h in selected})

    if args.optimizers is not None:
        found_names = {n.casefold() for h in selected
                       for n in (h["optimizer"], h["prefix"])}
        for requested in args.optimizers:
            if requested.casefold() not in found_names:
                print(f"Aviso: optimizador solicitado sin historiales: {requested}")

    indice = {(h["shots"], h["budget"], h["prefix"], h["run"]): h
              for h in selected}
    output_dir = args.output_dir or (scenario / "figures")
    if not output_dir.is_absolute():
        output_dir = BASE / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    optimo = cargar_optimo(scenario, args.no_optimum)
    summary = []
    produced = 0
    for n_shots in shots:
        for budget in budgets:
            fig, ax = plt.subplots(figsize=(10, 6))
            plotted = 0
            for prefix in prefixes:
                for run in runs:
                    key = (n_shots, budget, prefix, run)
                    h = indice.get(key)
                    if h is None:
                        print(f"Aviso: falta historial shots={n_shots}, budget={budget}, "
                              f"optimizer={prefix}, run={run}")
                        continue
                    df = cargar_historial(h["path"])
                    if df is None:
                        continue
                    y = "best_energy" if args.metric == "best" else "energy"
                    ax.plot(df["evaluation"], df[y],
                            label=f"{h['optimizer']} - ejecución {run}", linewidth=1.8)
                    summary.append({
                        "optimizer": h["optimizer"],
                        "file_prefix": prefix,
                        "shots": n_shots,
                        "budget": budget,
                        "run": run,
                        "evaluations": len(df),
                        "last_evaluation": df["evaluation"].iloc[-1],
                        "best_energy": float(df["best_energy"].iloc[-1]),
                        "final_energy": float(df["energy"].iloc[-1]),
                        "energy_gap": (float(df["best_energy"].iloc[-1]) - optimo)
                                      if optimo is not None else None,
                    })
                    plotted += 1

            if not plotted:
                plt.close(fig)
                continue

            if optimo is not None:
                ax.axhline(optimo, color="black", linestyle="--", linewidth=1.8,
                           label=f"Óptimo clásico = {optimo:.6f}")
            ax.set_xlabel("Evaluaciones de la función objetivo")
            ax.set_ylabel("Mejor energía observada" if args.metric == "best"
                          else "Energía observada")
            ax.set_title(f"Convergencia VQE-Dicke | {scenario.name} | {n_shots} shots\n"
                         f"Presupuesto configurado: {budget}")
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()

            # Cada selección produce un nombre distinto, para evitar colisiones habituales.
            opt_tag = "-".join(etiqueta_archivo(p) for p in prefixes)
            run_tag = "-".join(map(str, runs))
            filename = (f"convergence_shots{n_shots}_budget{budget}_"
                        f"opt-{opt_tag}_runs-{run_tag}_{args.metric}.png")
            ruta_salida = output_dir / filename
            fig.savefig(ruta_salida, dpi=300)
            plt.close(fig)
            print(f"Gráfica guardada: {ruta_salida}")
            produced += 1

    if summary:
        results = pd.DataFrame(summary)
        print("\nRESUMEN DE RESULTADOS:")
        print(results.to_string(index=False))
        selected_opt = "-".join(etiqueta_archivo(p) for p in prefixes)
        shots_tag = "-".join(map(str, shots))
        budget_tag = "-".join(map(str, budgets))
        runs_tag = "-".join(map(str, runs))
        csv = output_dir / (f"convergencia_shots-{shots_tag}_budget-{budget_tag}_"
                            f"opt-{selected_opt}_runs-{runs_tag}.csv")
        results.to_csv(csv, index=False)
        print(f"Resumen guardado: {csv}")
    print(f"\nGráficas generadas: {produced}")
    return 0 if produced else 1


if __name__ == "__main__":
    sys.exit(main())
