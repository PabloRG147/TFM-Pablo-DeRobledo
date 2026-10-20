#!/usr/bin/env python3
"""Metricas VQE para escenario I y escenarios multiclase II/III.

Correcciones respecto de metricas_scursulim.py:
- Factibilidad POR BLOQUE, no solo cardinalidad total.
- Aprovecha el ansatz de bloques independientes para evitar Statevector 2**25.
- Calcula Emax global exacto con una busqueda por mitades de un Hamiltoniano Ising cuadratico.

No repite entrenamientos. Los parametros evaluados son el mejor punto OBSERVADO
(no necesariamente el ultimo), y las probabilidades son ideales (sin shots).
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

BASE = Path(__file__).resolve().parent
PREFIX = {"COBYLA": "scipy_COBYLA", "CMA-ES": "CMA-ES"}


def quadratic_ising(ising, offset, n):
    """Energia = c + x @ l + x @ Q @ x; Q simetrica."""
    c = float(np.real(offset))
    if abs(float(np.imag(offset))) > 1e-9:
        raise ValueError("Offset complejo")
    lin = np.zeros(n)
    quad = np.zeros((n, n))
    for label, raw_coeff in ising.to_list():
        z = [j for j, ch in enumerate(reversed(label)) if ch == "Z"]
        if len(label) != n or any(ch not in "IZ" for ch in label):
            raise ValueError(f"Termino Ising no diagonal: {label}")
        coef = complex(raw_coeff)
        if abs(coef.imag) > 1e-9:
            raise ValueError("Coeficiente complejo")
        v = float(coef.real)
        c += v
        if len(z) == 1:
            lin[z[0]] -= 2 * v
        elif len(z) == 2:
            i, j = z
            lin[i] -= 2 * v
            lin[j] -= 2 * v
            quad[i, j] += 2 * v
            quad[j, i] += 2 * v
        elif len(z) > 2:
            raise ValueError("El maximo global rapido exige Ising cuadratico (I/Z/ZZ)")
    return c, lin, quad


def energies_from_bits(bits, c, lin, quad):
    return c + bits @ lin + np.einsum('bi,ij,bj->b', bits, quad, bits, optimize=True)


def all_binary(m):
    numbers = np.arange(1 << m, dtype=np.uint32)
    return ((numbers[:, None] >> np.arange(m, dtype=np.uint32)) & 1).astype(float)


def exact_global_extrema(c, lin, quad):
    """Evalua exactamente max/min de 2**n carteras sin materializar 2**n energias."""
    n = len(lin)
    split = n // 2
    left, right = all_binary(split), all_binary(n - split)
    eleft = energies_from_bits(left, 0.0, lin[:split], quad[:split, :split])
    eright = energies_from_bits(right, 0.0, lin[split:], quad[split:, split:])
    # Una sola matriz intermedia (right x split), independiente de 2**n.
    cross_right = 2 * (right @ quad[split:, :split])
    lo = float('inf')
    hi = float('-inf')
    for start in range(0, len(left), 32):
        stop = min(start + 32, len(left))
        vals = c + eleft[start:stop, None] + eright[None, :]
        vals = vals + left[start:stop] @ cross_right.T
        lo = min(lo, float(vals.min()))
        hi = max(hi, float(vals.max()))
    return lo, hi


def feasible_bases(n, group_sizes, weights):
    """Retorna indices computacionales y estados locales para cada cartera factible."""
    possibilities = []
    for m, k in zip(group_sizes, weights):
        possibilities.append([
            sum(1 << j for j in combination)
            for combination in itertools.combinations(range(m), k)
        ])
    states = np.asarray(list(itertools.product(*possibilities)), dtype=np.uint32)
    offsets = np.cumsum([0] + list(group_sizes[:-1]), dtype=np.uint32)
    indices = np.sum(states << offsets[None, :], axis=1, dtype=np.uint32)
    bitmat = ((indices[:, None] >> np.arange(n, dtype=np.uint32)) & 1).astype(float)
    return indices, states, bitmat


def bound_qc(qc, params):
    if not isinstance(params, dict):
        raise ValueError("Cada evaluacion requiere un diccionario de parametros")
    given = {str(k): float(np.asarray(v).item()) for k, v in params.items()}
    circuit_names = {str(p): p for p in qc.parameters}
    if len(given) != len(params) or given.keys() != circuit_names.keys():
        raise ValueError("Nombres de parametros incorrectos en historial")
    if not all(np.isfinite(v) for v in given.values()):
        raise ValueError("Parametros no finitos")
    return qc.assign_parameters({p: given[s] for s, p in circuit_names.items()})


def local_probabilities(qc, params, group_sizes, weights):
    """Estado por bloques sin crear el vector de amplitudes 2**25."""
    bounded = bound_qc(qc, params)
    offsets = np.cumsum([0] + list(group_sizes[:-1])).tolist()
    local_circuits = [QuantumCircuit(m) for m in group_sizes]
    for instruction in bounded.data:
        qargs = [bounded.find_bit(q).index for q in instruction.qubits]
        cargs = list(instruction.clbits)
        if cargs:
            raise ValueError("El circuito contiene medidas/instrucciones clasicas")
        if not qargs:
            # ignorar instrucciones globales sin qubits, si existieran
            continue
        groups = [i for i, (start, m) in enumerate(zip(offsets, group_sizes))
                  if all(start <= q < start + m for q in qargs)]
        if len(groups) != 1:
            raise ValueError("El ansatz entrelaza bloques: no se puede factorizar")
        group = groups[0]
        local_circuits[group].append(instruction.operation,
                                     [q - offsets[group] for q in qargs])
    local_prob = []
    leaks = []
    for circ, k in zip(local_circuits, weights):
        probs = np.abs(Statevector.from_instruction(circ).data) ** 2
        local_prob.append(probs)
        leaks.append(float(sum(probs[i] for i in range(len(probs)) if i.bit_count() != k)))
    return local_prob, max(leaks)


def prob_on_feasible(qc, params, group_sizes, weights, local_states):
    local_prob, leak = local_probabilities(qc, params, group_sizes, weights)
    if leak > 1e-8:
        raise ValueError(f"Estado con probabilidad inviable por bloque: {leak:.3g}")
    p = np.ones(len(local_states), dtype=float)
    for g, prob in enumerate(local_prob):
        p *= prob[local_states[:, g]]
    if not np.isclose(p.sum(), 1.0, atol=1e-8):
        raise ValueError(f"Las probabilidades factibles no suman uno: {p.sum()}")
    return p, leak


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', required=True)
    parser.add_argument('--shots', type=int, default=4096)
    parser.add_argument('--budget', type=int, default=1000)
    parser.add_argument('--optimizers', nargs='+', choices=list(PREFIX), default=list(PREFIX))
    parser.add_argument('--runs', nargs='+', type=int, default=list(range(10)))
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.shots <= 0 or args.budget <= 0 or not args.runs or min(args.runs) < 0 or len(set(args.runs)) != len(args.runs):
        parser.error('shots/budget positivos y runs unicos no negativos')
    scenario = Path(args.scenario).expanduser()
    if not scenario.is_absolute():
        scenario = BASE / scenario
    if not scenario.is_dir():
        raise FileNotFoundError(scenario)

    info = pd.read_pickle(scenario / 'scenario_info.pkl')
    meta = json.loads((scenario / 'metadata.json').read_text(encoding='utf-8'))
    qc = info['circuit']
    n = qc.num_qubits
    init = np.load(scenario / 'initial_points.npy')
    if max(args.runs) >= len(init) or init.shape[1] != qc.num_parameters:
        raise ValueError('Puntos iniciales incompatibles con los indices solicitados')
    if 'selected_per_class' in meta:
        weights = list(map(int, meta['selected_per_class']))
        group_sizes = [5] * len(weights)
    else:
        group_sizes = [n]
        weights = [int(meta.get('n_selected', info['target'].count('1')))]
    if sum(group_sizes) != n:
        raise ValueError('Los bloques no cubren el circuito')

    c, lin, quad = quadratic_ising(info['ising'], info['offset'], n)
    indices, local_states, bitmat = feasible_bases(n, group_sizes, weights)
    energies = energies_from_bits(bitmat, c, lin, quad)
    sorted_positions = np.argsort(energies, kind='stable')
    top5pos = sorted_positions[:5]
    exact_min_feasible = float(energies[sorted_positions[0]])
    e0 = float(info['optimal_obj_value'])
    target_index = int(info['target'], 2)
    where_target = np.flatnonzero(indices == target_index)
    if where_target.size != 1:
        raise ValueError('El target NO cumple las restricciones por clase')
    target_pos = int(where_target[0])
    if abs(e0 - exact_min_feasible) > 1e-8 or abs(energies[target_pos] - e0) > 1e-8:
        raise ValueError(f'Inconsistencia del optimo: almacenado={e0:.9f}, '
                         f'minimo factible={exact_min_feasible:.9f}, '
                         f'target={energies[target_pos]:.9f}')
    print(f'\nEscenario: {scenario.name} | qubits={n} | restricciones={weights}')
    print(f'Carteras factibles: {len(indices)} | optimo factible: {e0:.9f}')
    print('Calculando Emax global exacto (sin generar Statevector completo)...', flush=True)
    global_min, global_max = exact_global_extrema(c, lin, quad)
    feasible_max = float(np.max(energies))
    if global_max <= e0 or feasible_max <= e0:
        raise ValueError('Denominador del ratio de aproximacion no positivo')
    print(f'Emax global: {global_max:.9f} | Emax factible: {feasible_max:.9f}')
    print(f'Emin global: {global_min:.9f} | Emin factible: {e0:.9f}')

    initial_cache = {}
    for run in args.runs:
        params = {p: float(v) for p, v in zip(qc.parameters, init[run])}
        probs, _ = prob_on_feasible(qc, params, group_sizes, weights, local_states)
        initial_cache[run] = float(probs @ energies)

    data = []
    for opt in args.optimizers:
        for run in args.runs:
            filename = f'{PREFIX[opt]}_shots{args.shots}_budget{args.budget}_run{run}.pkl'
            path = scenario / 'results' / 'histories' / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            history = pd.read_pickle(path)
            history = history.copy()
            if not {'energy', 'evaluation', 'parameters'}.issubset(history.columns):
                raise ValueError(f'Columnas invalidas en {filename}')
            history['energy'] = pd.to_numeric(history['energy'], errors='coerce')
            history['evaluation'] = pd.to_numeric(history['evaluation'], errors='coerce')
            history = history.replace([np.inf, -np.inf], np.nan).dropna(subset=['energy', 'evaluation'])
            if history.empty:
                raise ValueError(f'Historial sin datos: {filename}')
            best = history.loc[history['energy'].idxmin()]
            probs, leak = prob_on_feasible(qc, best['parameters'], group_sizes, weights, local_states)
            energy = float(probs @ energies)
            pt = float(probs[target_pos])
            mp = float(np.max(probs))
            is_unique_mode = bool(pt >= mp - 1e-12 and np.count_nonzero(np.abs(probs - mp) < 1e-12) == 1)
            initial_e = initial_cache[run]
            record = {
                'optimizer': opt, 'run': run, 'shots': args.shots, 'budget': args.budget,
                'evaluations': len(history), 'best_evaluation': int(best['evaluation']),
                'initial_energy_exact': initial_e, 'final_energy_exact': energy,
                'best_observed_energy': float(best['energy']),
                'ar_paper_initial': (global_max-initial_e)/(global_max-e0),
                'ar_paper_final': (global_max-energy)/(global_max-e0),
                'ar_feasible_initial': (feasible_max-initial_e)/(feasible_max-e0),
                'ar_feasible_final': (feasible_max-energy)/(feasible_max-e0),
                'p_target': pt, 'is_target_unique_mode': int(is_unique_mode),
                'p_target_ge_095': int(pt >= 0.95),
                'success_with_p_ge_095': int(is_unique_mode and pt >= 0.95),
                'max_probability': mp,
                'mode_ties': int(np.count_nonzero(np.abs(probs - mp) < 1e-12)),
                'most_probable_bitstring': format(int(indices[int(np.argmax(probs))]), f'0{n}b'),
                'p_infeasible': leak,
            }
            for rank, pos in enumerate(top5pos, 1):
                record[f'p_top_{rank}'] = float(probs[pos])
            record['p_other_than_top5'] = float(1 - np.sum(probs[top5pos]))
            data.append(record)
        print(f'  Analizadas {len(args.runs)} ejecuciones de {opt}')

    df = pd.DataFrame(data)
    summary = df.groupby('optimizer', sort=False).agg(
        n=('run','size'), mean_ar_initial=('ar_paper_initial','mean'),
        mean_ar_final=('ar_paper_final','mean'), std_ar_final=('ar_paper_final','std'),
        median_ar_final=('ar_paper_final','median'), mean_p_target=('p_target','mean'),
        target_mode_count=('is_target_unique_mode','sum'),
        p_ge_095_count=('p_target_ge_095','sum'),
        mode_and_p_ge_095=('success_with_p_ge_095','sum'),
        mean_evaluations=('evaluations','mean'),
    ).reset_index()
    top_summary = df.groupby('optimizer', sort=False)[
        [f'p_top_{j}' for j in range(1, 6)] + ['p_other_than_top5']
    ].mean().reset_index()

    print('\nMETRICAS COMPARABLES CON ECUACION 7 / TABLA 3 (Statevector IDEAL)')
    print(summary.round(6).to_string(index=False))
    print('\nPROBABILIDAD MEDIA DE LAS 5 MEJORES CARTERAS FACTIBLES')
    print('Bitstrings:', ', '.join(format(int(indices[p]), f'0{n}b') for p in top5pos))
    print(top_summary.round(6).to_string(index=False))

    out = args.output_dir or (scenario / 'figures' / 'metricas_scursulim')
    if not out.is_absolute():
        out = BASE / out
    out.mkdir(parents=True, exist_ok=True)
    runs = sorted(args.runs)
    suffix = f'shots{args.shots}_budget{args.budget}_runs{runs[0]}-{runs[-1]}_n{len(runs)}'
    outputs = [out / f'detalle_{suffix}.csv', out / f'resumen_{suffix}.csv',
               out / f'top5_{suffix}.csv']
    df.to_csv(outputs[0], index=False)
    summary.to_csv(outputs[1], index=False)
    top_summary.to_csv(outputs[2], index=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5.5))
    xs, names, positions = [], [], []
    for i, opt in enumerate(args.optimizers):
        subset = df.loc[df.optimizer == opt]
        xs.extend([subset['ar_paper_initial'], subset['ar_paper_final']])
        positions.extend([3*i+1, 3*i+2])
        names.extend([f'{opt}\ninicial', f'{opt}\nfinal'])
    ax.boxplot(xs, positions=positions, widths=0.65, showmeans=True)
    ax.set_xticks(positions, names)
    ax.axhline(0.9, linestyle='--', linewidth=1, label='Ratio = 0.9')
    ax.axhline(1.0, linestyle=':', linewidth=1, label='Optimo = 1')
    ax.set_ylabel('Ratio de aproximacion (Emax global)')
    ax.set_title(f'{scenario.name} | {args.shots} shots | budget {args.budget}')
    ax.grid(axis='y', alpha=0.2)
    ax.legend()
    fig.tight_layout()
    plotpath = out / f'aproximacion_antes_despues_{suffix}.png'
    fig.savefig(plotpath, dpi=220)
    plt.close(fig)
    print('\nARCHIVOS:')
    for p in [*outputs, plotpath]:
        print(' ', p)
    print('\nNOTA: puntos seleccionados por energia observada minima, probabilidades ideales.')


if __name__ == '__main__':
    main()
