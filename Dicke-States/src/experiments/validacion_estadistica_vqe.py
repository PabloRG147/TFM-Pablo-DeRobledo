#!/usr/bin/env python3
"""Auditoria del bitstring optimo y comparaciones emparejadas de VQE.

No ejecuta nuevas optimizaciones. Relee historiales, reevalua el mejor punto
observado con Statevector, valida la codificacion del Hamiltoniano y calcula
estadisticas exploratorias emparejadas (IC bootstrap y test de signos).

Ejemplo (desde src/experiments):
    pipenv run python validar_estadistica_vqe.py \
        --scenario scursulim_feb_scenarioI --shots 256 4096 \
        --budget 300 --optimizers COBYLA CMA-ES --runs 0 1 2 3 4 5 6 7 8 9
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent
PATRON = re.compile(
    r"^(?P<prefix>.+)_shots(?P<shots>\d+)_budget(?P<budget>\d+)_run(?P<run>\d+)\.pkl$"
)


def opciones():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", required=True, help="Carpeta de escenario o ruta absoluta")
    p.add_argument("--shots", type=int, nargs="+", default=[256, 4096])
    p.add_argument("--budget", type=int, default=300)
    p.add_argument("--optimizers", nargs=2, default=["COBYLA", "CMA-ES"],
                   metavar=("OPT_A", "OPT_B"), help="Los dos optimizadores a comparar")
    p.add_argument("--runs", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--bootstrap", type=int, default=20000,
                   help="Numero de remuestreos bootstrap emparejados")
    p.add_argument("--seed", type=int, default=20261018,
                   help="Semilla para los intervalos bootstrap")
    p.add_argument("--output-dir", type=Path, default=None)
    a = p.parse_args()
    if (a.budget <= 0 or any(s <= 0 for s in a.shots)
            or any(r < 0 for r in a.runs) or a.bootstrap < 100):
        p.error("budget y shots deben ser positivos, runs no negativos y bootstrap >= 100")
    if len(set(a.runs)) != len(a.runs) or len(set(a.shots)) != len(a.shots):
        p.error("No repitas indices de runs ni de shots")
    if a.optimizers[0].casefold() == a.optimizers[1].casefold():
        p.error("Debes seleccionar dos optimizadores distintos")
    return a


def energia_diagonal(ising, offset, n):
    """H|z> a partir de etiquetas Pauli. Qubit 0 = caracter derecho del Pauli."""
    posiciones = np.arange(1 << n, dtype=np.uint32)
    energias = np.full(1 << n, float(np.real(complex(offset))), dtype=float)
    if abs(complex(offset).imag) > 1e-10:
        raise ValueError("Offset con parte imaginaria no nula")
    for etiqueta, c in ising.to_list():
        if len(etiqueta) != n or set(etiqueta) - {"I", "Z"}:
            raise ValueError(f"Hamiltoniano no diagonal o etiqueta invalida: {etiqueta}")
        if abs(complex(c).imag) > 1e-10:
            raise ValueError("Coeficiente del Hamiltoniano con parte imaginaria no nula")
        paridad = np.zeros(len(posiciones), dtype=np.int8)
        for qubit, signo in enumerate(reversed(etiqueta)):
            if signo == "Z":
                paridad ^= ((posiciones >> qubit) & 1).astype(np.int8)
        energias += float(np.real(c)) * (1 - 2 * paridad)
    return energias


def comprobar_objetivo(info, metadata):
    from qiskit.quantum_info import Statevector

    qc, ising = info["circuit"], info["ising"]
    n = qc.num_qubits
    if n != ising.num_qubits or qc.num_clbits:
        raise ValueError("Numero de qubits inconsistente, o circuito con mediciones")
    objetivo = info["target"]
    if not isinstance(objetivo, str) or len(objetivo) != n or set(objetivo) - {"0", "1"}:
        raise ValueError(f"target no es un bitstring de longitud {n}: {objetivo!r}")
    k = int(metadata.get("n_selected", objetivo.count("1")))
    if objetivo.count("1") != k:
        raise ValueError(f"target selecciona {objetivo.count('1')} activos, se esperaban {k}")

    energia = energia_diagonal(ising, info["offset"], n)
    factibles = np.array([i for i in range(1 << n) if i.bit_count() == k], dtype=int)
    minimo = float(np.min(energia[factibles]))
    atol = 1e-8
    optimos = factibles[np.abs(energia[factibles] - minimo) <= atol]
    idx = int(objetivo, 2)  # Qiskit imprime q[n-1]...q[0]
    if idx not in optimos:
        ejemplos = [format(i, f"0{n}b") for i in optimos[:5]]
        raise ValueError(f"Bitstring target {objetivo} NO es optimo. Optimos: {ejemplos}")
    if not np.isclose(minimo, float(info["optimal_obj_value"]), atol=atol, rtol=0):
        raise ValueError("La energia minima factible del Ising no coincide con optimal_obj_value")
    estado_base = Statevector.from_int(idx, dims=2**n)
    comprobacion = complex(estado_base.expectation_value(ising)) + complex(info["offset"])
    if abs(comprobacion.imag) > atol or abs(comprobacion.real - energia[idx]) > atol:
        raise ValueError("Posible inversion de bits: Statevector y diagonal calculada no concuerdan")

    print("\nVALIDACION DEL PROBLEMA")
    print(f"  Qubits: {n}; seleccionados: {k}; carteras factibles: {len(factibles)}")
    print(f"  target almacenado: {objetivo} (indice base 2: {idx})")
    print(f"  minimo Ising factible: {minimo:.9f}")
    print(f"  optimo del escenario: {float(info['optimal_obj_value']):.9f}")
    print(f"  carteras empatadas en el optimo (tolerancia {atol}): {len(optimos)}")
    otras = energia[factibles][energia[factibles] > minimo + atol]
    if otras.size:
        siguiente = float(np.min(otras))
        print(f"  siguiente energia factible: {siguiente:.9f}; separacion: {siguiente - minimo:.9f}")
    print("  VALIDADO: orden de bits y target coherentes con el minimo del Hamiltoniano")
    if len(optimos) > 1:
        print("  AVISO: existen multiples carteras optimas. Se calcularan p(target) y p(cualquier optimo).")
    return energia, optimos, k


def preparar_estado(circuito, parametros):
    """Reconstruir estado a partir del dict str->float (SciPy) o Parameter->float (Optuna)."""
    from qiskit.quantum_info import Statevector

    if not isinstance(parametros, dict):
        raise ValueError("El historial no almacena los parametros en un diccionario")
    nombres = {str(p): p for p in circuito.parameters}
    guardados = {str(p): float(np.asarray(v).item()) for p, v in parametros.items()}
    if len(guardados) != len(parametros):
        raise ValueError("Hay parametros duplicados al convertir los nombres a cadenas")
    if nombres.keys() != guardados.keys():
        raise ValueError(f"Parametros no coinciden: faltan {set(nombres)-set(guardados)}; "
                         f"sobran {set(guardados)-set(nombres)}")
    if not all(np.isfinite(x) for x in guardados.values()):
        raise ValueError("Parametros no finitos")
    enlazados = circuito.assign_parameters({p: guardados[s] for s, p in nombres.items()})
    return Statevector.from_instruction(enlazados), guardados


def leer_resultado(ruta, info, energias, optimos, k, run, iniciales):
    historial = pd.read_pickle(ruta)
    if not isinstance(historial, pd.DataFrame) or not {"energy", "evaluation", "parameters"}.issubset(historial.columns):
        raise ValueError(f"Formato de historial invalido: {ruta.name}")
    historial = historial.copy()
    historial["energy"] = pd.to_numeric(historial["energy"], errors="coerce")
    historial["evaluation"] = pd.to_numeric(historial["evaluation"], errors="coerce")
    historial = historial.replace([np.inf, -np.inf], np.nan).dropna(subset=["energy", "evaluation"])
    if historial.empty:
        raise ValueError(f"Historial sin valores numericos: {ruta.name}")
    historial = historial.sort_values("evaluation")
    mejor = historial.loc[historial["energy"].idxmin()]
    estado, _ = preparar_estado(info["circuit"], mejor["parameters"])
    prob = np.abs(np.asarray(estado.data))**2
    if not np.isclose(prob.sum(), 1, atol=1e-9):
        raise ValueError(f"Estado no normalizado: {ruta.name}")
    fuga = sum(prob[i] for i in range(len(prob)) if i.bit_count() != k)
    if fuga > 1e-7:
        raise ValueError(f"{ruta.name}: probabilidad inviable {fuga}")
    energia_exacta = float(np.dot(prob, energias))
    energia_qiskit = complex(estado.expectation_value(info["ising"])) + complex(info["offset"])
    if abs(energia_qiskit.imag) > 1e-8 or abs(energia_qiskit.real - energia_exacta) > 1e-8:
        raise ValueError(f"{ruta.name}: incoherencia entre energia Statevector y diagonal")
    objetivo = info["target"]
    prob_objetivo = float(prob[int(objetivo, 2)])
    p_dict = float(estado.probabilities_dict().get(objetivo, 0.0))
    if abs(prob_objetivo - p_dict) > 1e-9:
        raise ValueError(f"{ruta.name}: inconsistencia en indice de target y probabilidades_dict")
    prob_todos = float(np.sum(prob[optimos]))
    if energia_exacta < float(info["optimal_obj_value"]) - 1e-7:
        raise ValueError(f"{ruta.name}: energia exacta inferior al optimo factible")

    # Auditoria informativa: el primer punto evaluado no tiene por que ser la
    # media inicial de CMA-ES; NO se interpreta como error automatico.
    primera = historial.iloc[0]["parameters"]
    _, primer_mapa = preparar_estado(info["circuit"], primera)
    primer_vector = np.array([primer_mapa[str(p)] for p in info["circuit"].parameters])
    inicial_vector = iniciales[run]
    distancia_inicio = float(np.max(np.abs(primer_vector - inicial_vector)))

    return {
        "run": run,
        "evaluations": len(historial),
        "best_evaluation": int(mejor["evaluation"]),
        "best_observed": float(mejor["energy"]),
        "exact_energy": energia_exacta,
        "p_target": prob_objetivo,
        "p_any_optimum": prob_todos,
        "infeasible_p": float(fuga),
        "first_point_max_abs_diff": distancia_inicio,
    }


def par_bootstrap(diferencias, rng, n_bootstrap):
    """IC percentil del promedio de diferencias manteniendo el emparejamiento."""
    diferencias = np.asarray(diferencias, dtype=float)
    n = len(diferencias)
    if n < 2:
        return (float("nan"), float("nan"))
    indexes = rng.integers(0, n, size=(n_bootstrap, n))
    medias = diferencias[indexes].mean(axis=1)
    return tuple(np.percentile(medias, [2.5, 97.5]).tolist())


def informe_pareado(df, col, grupo, primero, segundo, rng, n_bootstrap, etiqueta):
    seleccion = df.pivot(index="run", columns=grupo, values=col)
    if primero not in seleccion.columns or segundo not in seleccion.columns:
        raise ValueError(f"Falta pareja en {etiqueta}")
    seleccion = seleccion[[primero, segundo]].dropna()
    d = (seleccion[primero] - seleccion[segundo]).to_numpy(dtype=float)
    wins_a = int(np.count_nonzero(d < -1e-10))
    wins_b = int(np.count_nonzero(d > 1e-10))
    ties = int(len(d) - wins_a - wins_b)
    n_eff = wins_a + wins_b
    p = float(binomtest(wins_a, n_eff, 0.5).pvalue) if n_eff > 0 else 1.0
    low, high = par_bootstrap(d, rng, n_bootstrap)
    return {
        "comparison": etiqueta,
        "metric": col,
        "a": str(primero), "b": str(segundo),
        "n_paired": len(d), "mean_delta_a_minus_b": float(d.mean()),
        "median_delta_a_minus_b": float(np.median(d)),
        "ci95_boot_low": low, "ci95_boot_high": high,
        "a_lower_count": wins_a, "b_lower_count": wins_b,
        "ties": ties, "sign_test_p_two_sided": p,
    }


def main():
    a = opciones()
    escenario = Path(a.scenario).expanduser()
    if not escenario.is_absolute():
        escenario = ROOT / escenario
    escenario = escenario.resolve()
    archivo_info = escenario / "scenario_info.pkl"
    if not archivo_info.exists():
        raise FileNotFoundError(archivo_info)
    info = pd.read_pickle(archivo_info)
    meta_path = escenario / "metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    iniciales = np.load(escenario / "initial_points.npy")
    if iniciales.ndim != 2 or iniciales.shape[1] != info["circuit"].num_parameters:
        raise ValueError("La forma de initial_points.npy no coincide con el ansatz")
    if max(a.runs) >= len(iniciales):
        raise ValueError(f"Solicitado run {max(a.runs)}, pero solo hay {len(iniciales)} inicializaciones")
    energias, optimos, k = comprobar_objetivo(info, metadata)
    carpeta = escenario / "results" / "histories"
    if not carpeta.is_dir():
        raise FileNotFoundError(carpeta)

    # Descubrimiento por nombre real; no presuponemos prefijos fuera de scipy_/optuna_.
    ubicaciones = {}
    for ruta in carpeta.glob("*.pkl"):
        match = PATRON.fullmatch(ruta.name)
        if not match:
            continue
        m = match.groupdict()
        prefijo = m["prefix"]
        nombre = prefijo
        for comienzo in ("scipy_", "optuna_"):
            if prefijo.startswith(comienzo):
                nombre = prefijo[len(comienzo):]
                break
        key = (nombre.casefold(), int(m["shots"]), int(m["budget"]), int(m["run"]))
        if key in ubicaciones:
            raise ValueError(f"Dos historiales encontrados para {key}")
        ubicaciones[key] = ruta

    registros = []
    for shots in a.shots:
        for opt in a.optimizers:
            for run in a.runs:
                key = (opt.casefold(), shots, a.budget, run)
                ruta = ubicaciones.get(key)
                if ruta is None:
                    raise FileNotFoundError(f"Falta historial: {opt}, shots={shots}, budget={a.budget}, run={run}")
                fila = leer_resultado(ruta, info, energias, optimos, k, run, iniciales)
                fila.update(shots=shots, budget=a.budget, optimizer=opt, file=ruta.name)
                registros.append(fila)

    df = pd.DataFrame(registros)
    print(f"\nREEVALUACIONES: {len(df)} historiales comprobados")
    resumen = df.groupby(["shots", "optimizer"]).agg(
        n=("run", "size"), mean_energy=("exact_energy", "mean"),
        std_energy=("exact_energy", "std"),
        median_energy=("exact_energy", "median"),
        mean_p_target=("p_target", "mean"),
        mean_p_any_optimum=("p_any_optimum", "mean"),
        max_infeasible_p=("infeasible_p", "max"),
        mean_first_point_diff=("first_point_max_abs_diff", "mean"),
    ).reset_index()
    print("\nESTADISTICAS Y PROBABILIDAD OPTIMA (sin ruido):")
    print(resumen.round(8).to_string(index=False))
    print("\nAUDITORIA DEL PRIMER PUNTO:")
    print("  first_point_diff compara la primera evaluacion con initial_points[run].")
    print("  En CMA-ES la primera propuesta puede diferir de su centro inicial;")
    print("  esto NO demuestra por si solo que las inicializaciones sean distintas.")

    rng = np.random.default_rng(a.seed)
    comparaciones = []
    opt_a, opt_b = a.optimizers
    for shots in a.shots:
        bloque = df.loc[df.shots == shots]
        for col in ("exact_energy", "p_target"):
            comparaciones.append(informe_pareado(
                bloque, col, "optimizer", opt_a, opt_b, rng, a.bootstrap,
                f"{shots} shots: {opt_a} - {opt_b}"))
    if len(a.shots) == 2:
        for opt in a.optimizers:
            bloque = df.loc[df.optimizer == opt]
            comparaciones.append(informe_pareado(
                bloque, "exact_energy", "shots", a.shots[1], a.shots[0],
                rng, a.bootstrap, f"{opt}: {a.shots[1]} - {a.shots[0]} shots"))

    pares = pd.DataFrame(comparaciones)
    print("\nCOMPARACIONES EMPAREJADAS (delta = A - B):")
    for r in pares.itertuples():
        sentido = "menor energia = mejor" if r.metric == "exact_energy" else "mayor probabilidad = mejor"
        print(f"  {r.comparison} [{r.metric}]:")
        print(f"    n={r.n_paired}; delta medio={r.mean_delta_a_minus_b:+.6f}; "
              f"IC bootstrap 95%=[{r.ci95_boot_low:+.6f}, {r.ci95_boot_high:+.6f}]")
        print(f"    A menor: {r.a_lower_count}; B menor: {r.b_lower_count}; "
              f"empates: {r.ties}; test de signos bilateral p={r.sign_test_p_two_sided:.4f} ({sentido})")

    destino = a.output_dir or (escenario / "figures" / "validacion")
    if not destino.is_absolute():
        destino = ROOT / destino
    destino.mkdir(parents=True, exist_ok=True)
    etiqueta = f"budget{a.budget}_shots{'-'.join(map(str,a.shots))}_runs{'-'.join(map(str,a.runs))}"
    paths = [destino / f"detalle_validado_{etiqueta}.csv",
             destino / f"resumen_validado_{etiqueta}.csv",
             destino / f"comparaciones_pareadas_{etiqueta}.csv"]
    df.to_csv(paths[0], index=False)
    resumen.to_csv(paths[1], index=False)
    pares.to_csv(paths[2], index=False)
    print("\nARCHIVOS GENERADOS:")
    for p in paths:
        print(" ", p)
    print("\nNOTAS: pruebas exploratorias con n pequeno; no certifican superioridad estadistica.")
    print("Las comparaciones por run asumen inicializaciones comparables: revisar el uso de")
    print("initial_points en cada sampler/optimizador antes de interpretarlas causalmente.")


if __name__ == "__main__":
    main()
