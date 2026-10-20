#!/usr/bin/env python3
"""Añade CLI configurable a los scripts SciPy y Optuna sin borrar sus personalizaciones.

Uso desde src/experiments:
  python adaptar_cli_experimentos.py

Hace copias .antes_cli.py y comprueba AST de ambos scripts antes de escribir.
No ejecuta simulaciones.
"""

import ast
from pathlib import Path
import re
import sys

SCRIPTS = {
    "vqe_dicke_scipy_experiments.py": ("scipy", "scipy_{OPTIMIZER}"),
    "vqe_dicke_optuna_experiments.py": ("optuna", "{OPTIMIZER}"),
}

ARGUMENTOS = '''parser.add_argument("--shots", type=int, nargs="+", default=[256],
                    help="Número(s) de shots; ejemplo: --shots 256 4096")
parser.add_argument("--budgets", type=int, nargs="+", default=[100, 300],
                    help="Presupuestos maxiter/n_trials; ejemplo: --budgets 100 300")
parser.add_argument("--runs", type=int, nargs="+", default=[0, 1],
                    help="Índices de puntos iniciales; ejemplo: --runs 0 1")
parser.add_argument("--overwrite", action="store_true",
                    help="Permitir sobrescribir historiales/resultados existentes")
'''

# Las rutas de historiales actuales son fijas y las usa plot_historial.py.
CONFIG = '''N_SHOTS_LIST = list(dict.fromkeys(args.shots))
MAXITER_LIST = list(dict.fromkeys(args.budgets))
RUNS = list(dict.fromkeys(args.runs))
if any(s <= 0 for s in N_SHOTS_LIST):
    parser.error("--shots requiere enteros positivos")
if any(b <= 0 for b in MAXITER_LIST):
    parser.error("--budgets requiere enteros positivos")
if not RUNS or any(r < 0 or r >= len(initial_points) for r in RUNS):
    parser.error(
        f"--runs debe contener índices entre 0 y {len(initial_points)-1}; "
        "regenera initial_points.npy con más puntos si los necesitas"
    )

def _etiqueta_indices(valores):
    if len(valores) > 12:
        import hashlib
        sufijo = hashlib.sha256(",".join(map(str, valores)).encode()).hexdigest()[:8]
        return f"{min(valores)}-{max(valores)}-{sufijo}"
    return "-".join(map(str, valores))

RESULTS_TAG = f"budgets_{_etiqueta_indices(MAXITER_LIST)}_runs_{_etiqueta_indices(RUNS)}"
_results_dir = Path(SCENARIO) / "results" / "__KIND__"
_histories_dir = Path(SCENARIO) / "results" / "histories"
_results_dir.mkdir(parents=True, exist_ok=True)
_histories_dir.mkdir(parents=True, exist_ok=True)

# Comprobar ANTES de gastar tiempo en optimizaciones.
if not args.overwrite:
    conflictos = []
    for _shots in N_SHOTS_LIST:
        _result_file = _results_dir / f"{OPTIMIZER}_n_shots_{_shots}_{RESULTS_TAG}_experiments.pkl"
        if _result_file.exists():
            conflictos.append(_result_file)
        for _budget in MAXITER_LIST:
            for _run in RUNS:
                _history_file = _histories_dir / (
                    f"__HISTORY_PREFIX___shots{_shots}_budget{_budget}_run{_run}.pkl"
                )
                if _history_file.exists():
                    conflictos.append(_history_file)
    if conflictos:
        parser.error(
            "Ya existen resultados/historiales de estas configuraciones. "
            "No se sobrescriben automáticamente. Para repetirlas intencionadamente, "
            "añade --overwrite. Primer conflicto: " + str(conflictos[0])
        )

print(f"Configuración CLI: shots={N_SHOTS_LIST}, budgets={MAXITER_LIST}, runs={RUNS}")
'''


def one_sub(text, pattern, replacement, what, flags=re.MULTILINE):
    result, count = re.subn(pattern, replacement, text, count=1, flags=flags)
    if count != 1:
        raise ValueError(f"No he localizado con seguridad: {what}")
    return result


def transform(text, kind, prefix):
    if "RESULTS_TAG" in text or "--overwrite" in text:
        raise ValueError("Este archivo parece estar adaptado ya. No se aplican cambios.")

    if not re.search(r"(?m)^from pathlib import Path\b", text):
        text = one_sub(text, r"(?m)^import sys$", "import sys\nfrom pathlib import Path", "import sys")

    # Conservar parser y las opciones específicas ya existentes.
    text = one_sub(text, r'(?m)^args\s*=\s*parser\.parse_args\(\)\s*$',
                   ARGUMENTOS + 'args = parser.parse_args()', "parser.parse_args()")

    # Fundamental: NO cortar [:2] si en adelante se quieren runs 0..99.
    text = one_sub(text,
                   r'(?m)^(\s*)initial_points\s*=\s*np\.load\(f["\']\{SCENARIO\}/initial_points\.npy["\']\)(?:\[:\d+\])?\s*$',
                   r'\1initial_points = np.load(f"{SCENARIO}/initial_points.npy")',
                   "carga de initial_points")
    text = one_sub(text, r'(?m)^(\s*)NUM_EXPERIMENTS\s*=.*initial_points\.shape\[0\].*$',
                   r'\1NUM_EXPERIMENTS = len(initial_points)', "NUM_EXPERIMENTS")

    text = one_sub(text, r'(?m)^N_SHOTS_LIST\s*=\s*[^\n]+$',
                   '### N_SHOTS_LIST se define mediante --shots ###', "N_SHOTS_LIST")
    block = CONFIG.replace("__KIND__", kind).replace("__HISTORY_PREFIX__", prefix)
    text = one_sub(text, r'(?m)^MAXITER_LIST\s*=\s*[^\n]+$',
                   block.rstrip(), "MAXITER_LIST")

    text = one_sub(text,
                   r'for i in tqdm\(range\(NUM_EXPERIMENTS\),\s*desc="experiments",\s*leave=False\):',
                   'for i in tqdm(RUNS, desc="experiments", leave=False):',
                   "bucle de runs")

    # Registro explícito del límite configurado, distinto de nfev real en SciPy.
    text = one_sub(text, r'(?m)^(\s*)"n_iter",\s*$',
                   r'\1"n_iter",\n\1"budget",', "columna n_iter")
    if kind == "scipy":
        record_pattern = r'(?m)^(\s*)result_scipy\.nfev,\s*$'
        record_repl = r'\1result_scipy.nfev,\n\1maxiter,'
    else:
        # Dentro de data.append([ ... ]), justo después de n_shots.
        record_pattern = r'(?m)^(\s*)n_shots,\s*\n\s*maxiter,\s*$'
        record_repl = r'\1n_shots,\n\1maxiter,\n\1maxiter,'
    text = one_sub(text, record_pattern, record_repl, "registro por presupuesto")

    # Prefijo compatible con los programas de análisis que siguen leyendo histories/.
    suffix_old = '{OPTIMIZER}_n_shots_{n_shots}_experiments.pkl'
    suffix_new = '{OPTIMIZER}_n_shots_{n_shots}_{RESULTS_TAG}_experiments.pkl'
    if text.count(suffix_old) != 1:
        raise ValueError("No reconozco un único nombre de archivo resumen .pkl")
    text = text.replace(suffix_old, suffix_new)

    ast.parse(text)
    return text


def main():
    folder = Path(__file__).resolve().parent
    changed = {}
    for name, (kind, prefix) in SCRIPTS.items():
        path = folder / name
        if not path.is_file():
            raise FileNotFoundError(f"No encuentro {path}. Pon este parcheador junto a ambos scripts")
        original = path.read_text(encoding="utf-8")
        changed[name] = (original, transform(original, kind, prefix))

    # Ningún script se cambia hasta que AMBOS han pasado las comprobaciones.
    for name, (old, new) in changed.items():
        backup = folder / name.replace(".py", ".antes_cli.py")
        if backup.exists():
            raise FileExistsError(f"Ya existe {backup}; no sobrescribo copias de seguridad")
    for name, (old, new) in changed.items():
        path = folder / name
        backup = folder / name.replace(".py", ".antes_cli.py")
        backup.write_text(old, encoding="utf-8")
        path.write_text(new, encoding="utf-8")
        print(f"Adaptado: {name} (copia: {backup.name})")
    print("Listo. Revisa con pipenv run python <script>.py --help")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
