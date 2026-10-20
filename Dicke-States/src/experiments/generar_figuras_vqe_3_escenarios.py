#!/usr/bin/env python3
"""Figuras comparativas de VQE para tres escenarios de Scursulim.

Consume los CSV existentes generados con metricas_scursulim.py (I)
y metricas_scursulim_corregidas.py (II y III). No entrena circuitos.

Ejemplo, desde src/experiments:
  pipenv run python generar_figuras_vqe_3_escenarios.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCENARIOS = {
    "I": "scursulim_feb_scenarioI_real",
    "II": "scursulim_feb_scenarioII_real",
    "III": "scursulim_feb_scenarioIII_real",
}
OPTIMIZERS = ("COBYLA", "CMA-ES")
COLORS = {"COBYLA": "#C77832", "CMA-ES": "#2178A3"}
REQUIRED_SUMMARY = {
    "optimizer", "n", "mean_ar_initial", "mean_ar_final", "std_ar_final",
    "mean_p_target", "target_mode_count", "p_ge_095_count", "mean_evaluations",
}
REQUIRED_TOP5 = {"optimizer", *(f"p_top_{i}" for i in range(1, 6)), "p_other_than_top5"}


def read_csv(folder: Path, kind: str, shots: int, budget: int, start: int, n: int) -> pd.DataFrame:
    filename = f"{kind}_shots{shots}_budget{budget}_runs{start}-{start+n-1}_n{n}.csv"
    path = folder / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"No se ha encontrado: {path}\n"
            f"Genera primero las metricas de este escenario y verifica los indices."
        )
    df = pd.read_csv(path)
    required = REQUIRED_SUMMARY if kind == "resumen" else REQUIRED_TOP5
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path}: faltan columnas: {sorted(missing)}")
    if len(df) != len(OPTIMIZERS) or set(df["optimizer"]) != set(OPTIMIZERS):
        raise ValueError(f"{path}: se esperan exactamente una fila de COBYLA y CMA-ES")
    if kind == "resumen" and not (df["n"].astype(int) == n).all():
        raise ValueError(f"{path}: el numero n de inicializaciones no coincide con {n}")
    return df


def collect(base: Path, shots: int, budget: int, start: int, n: int):
    summaries, tops = [], []
    for key, scenario_dir in SCENARIOS.items():
        folder = base / scenario_dir / "figures" / "metricas_scursulim"
        for kind, accum in [("resumen", summaries), ("top5", tops)]:
            df = read_csv(folder, kind, shots, budget, start, n)
            df.insert(0, "scenario", key)
            accum.append(df)
    summary = pd.concat(summaries, ignore_index=True)
    top5 = pd.concat(tops, ignore_index=True)
    return summary, top5


def val(df: pd.DataFrame, scenario: str, opt: str, key: str) -> float:
    matched = df[(df["scenario"] == scenario) & (df["optimizer"] == opt)]
    return float(matched.iloc[0][key])


def stylize(ax, ylabel=None):
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18)
    if ylabel:
        ax.set_ylabel(ylabel)


def fig_ratio(summary: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    x = np.arange(len(OPTIMIZERS))
    w = 0.34
    for ax, scenario in zip(axes, SCENARIOS):
        ini = [val(summary, scenario, opt, "mean_ar_initial") for opt in OPTIMIZERS]
        fin = [val(summary, scenario, opt, "mean_ar_final") for opt in OPTIMIZERS]
        err = [val(summary, scenario, opt, "std_ar_final") for opt in OPTIMIZERS]
        b1 = ax.bar(x - w/2, ini, width=w, label="Antes", color="#9FAAB6")
        b2 = ax.bar(x + w/2, fin, width=w, yerr=err, capsize=3, label="Despues",
                    color="#2178A3", error_kw={"elinewidth": 1, "capthick": 1})
        ax.set_title(f"Escenario {scenario}", weight="bold")
        ax.set_xticks(x, OPTIMIZERS)
        ax.set_ylim(0, 1.10)
        stylize(ax, "Ratio de aproximacion" if scenario == "I" else None)
        for bar, value in zip(b2, fin):
            ax.text(bar.get_x() + bar.get_width()/2, min(1.09, value + 0.045),
                    f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.035), ncol=2, frameon=False)
    fig.suptitle("Aproximacion energetica antes y despues de VQE", weight="bold", fontsize=14)
    fig.text(0.5, -0.025, "Barras de error: desviacion tipica entre inicializaciones (solo valor final)",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    fig.savefig(out / "fig5_aproximacion_tres_escenarios.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_top5(top5: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    x = np.arange(1, 6)
    w = 0.36
    for ax, scenario in zip(axes, SCENARIOS):
        for oi, opt in enumerate(OPTIMIZERS):
            ys = [val(top5, scenario, opt, f"p_top_{i}") for i in x]
            positions = x + (oi - 0.5)*w
            ax.bar(positions, ys, width=w, color=COLORS[opt], label=opt)
        ax.set_xticks(x, [f"Top {i}" for i in x])
        ax.set_ylim(0, 1.07)
        ax.set_title(f"Escenario {scenario}", weight="bold")
        stylize(ax, "Probabilidad media ideal" if scenario == "I" else None)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.025), ncol=2, frameon=False)
    fig.suptitle("Probabilidad de las cinco carteras factibles de menor energia", weight="bold", fontsize=14)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(out / "fig6_top5_tres_escenarios.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def wilson_interval(k: int, n: int, z: float = 1.959963984540054):
    p = k/n
    den = 1 + z*z/n
    ctr = (p + z*z/(2*n))/den
    half = z/den * np.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return max(0.0, ctr-half), min(1.0, ctr+half)


def fig_success(summary: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    metrics = (("target_mode_count", "Optimo como cartera mas probable"),
               ("p_ge_095_count", "Probabilidad del optimo >= 95 %"))
    x = np.arange(3)
    w = 0.34
    for ax, (metric, title) in zip(axes, metrics):
        for oi, opt in enumerate(OPTIMIZERS):
            vals = []
            lower, upper, counts = [], [], []
            for scenario in SCENARIOS:
                n = int(val(summary, scenario, opt, "n"))
                count = int(val(summary, scenario, opt, metric))
                lo, hi = wilson_interval(count, n)
                p = count/n
                vals.append(p)
                lower.append(max(0.0, p-lo))
                upper.append(max(0.0, hi-p))
                counts.append((count,n))
            positions = x + (oi-0.5)*w
            ax.bar(positions, vals, width=w, color=COLORS[opt], label=opt)
            ax.errorbar(positions, vals, yerr=np.array([lower, upper]), fmt="none",
                        ecolor="#333333", capsize=3, elinewidth=1)
            for xp, p, (count,n) in zip(positions, vals, counts):
                ax.annotate(f"{count}/{n}", (xp, p), xytext=(0, 8), textcoords="offset points",
                            ha="center", va="bottom", fontsize=9, weight="bold")
        ax.set_title(title, weight="bold")
        ax.set_xticks(x, [f"Escenario {scenario}" for scenario in SCENARIOS])
        ax.set_ylim(0, 1.25)
        stylize(ax, "Frecuencia de ejecuciones" if metric == "target_mode_count" else None)
        ax.set_yticks(np.linspace(0,1,6), [f"{v:.0%}" for v in np.linspace(0,1,6)])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
    fig.suptitle("Frecuencia de identificacion del optimo (intervalo Wilson 95 %)",
                 weight="bold", fontsize=14)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(out / "frecuencias_exito_tres_escenarios.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_target(summary: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(3)
    w = 0.34
    for oi, opt in enumerate(OPTIMIZERS):
        ys = [val(summary, scenario, opt, "mean_p_target") for scenario in SCENARIOS]
        bars = ax.bar(x + (oi-0.5)*w, ys, width=w, label=opt, color=COLORS[opt])
        for bar, y in zip(bars, ys):
            ax.annotate(f"{100*y:.2f}%", (bar.get_x()+bar.get_width()/2, bar.get_height()),
                        xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(x, [f"Escenario {scenario}" for scenario in SCENARIOS])
    ax.set_ylim(0, 1.14)
    ax.set_yticks(np.linspace(0,1,6), [f"{v:.0%}" for v in np.linspace(0,1,6)])
    ax.set_title("Probabilidad media de medir la cartera optima", weight="bold", fontsize=13)
    stylize(ax, "Probabilidad media ideal")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "probabilidad_optimo_tres_escenarios.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parent,
                        help="Directorio src/experiments que contiene los escenarios")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n debe ser positivo")
    base = args.base.resolve()
    out = args.out.resolve() if args.out is not None else base / "figuras_vqe_3_escenarios"
    summary, top5 = collect(base, args.shots, args.budget, args.start, args.n)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "tabla3_parcial_vqe.csv", index=False)
    top5.to_csv(out / "top5_comparacion_vqe.csv", index=False)
    fig_ratio(summary, out)
    fig_top5(top5, out)
    fig_target(summary, out)
    fig_success(summary, out)
    view = summary[["scenario", "optimizer", "n", "mean_ar_initial", "mean_ar_final",
                    "mean_p_target", "target_mode_count", "p_ge_095_count", "mean_evaluations"]]
    print("\nRESUMEN DE LOS TRES ESCENARIOS:\n" + view.round(6).to_string(index=False))
    print("\nFIGURAS Y CSV GENERADOS:")
    for p in sorted(out.iterdir()):
        print(" ", p)
    print("\nNota: datos alternativos al paper, n=" + str(args.n) +
          "; se selecciona el mejor punto OBSERVADO y se evalua con Statevector ideal.")

if __name__ == "__main__":
    main()
