"""Prepara datos financieros auditables para reproducir metodológicamente
los escenarios multiclase II/III de Scursulim (2026).

NO son necesariamente los activos originales del artículo. Para esta
instancia se emplean acciones, participaciones de REIT, criptomonedas,
ETFs de bonos y ETFs/ETPs de materias primas como proxies.

Ejemplo, desde src/experiments:
    pipenv run python preparar_datos_multiclase.py --period 2

Produce ../datasets/scursulim_multiclase_p2_prices.csv,
../datasets/scursulim_multiclase_p2_classes.csv y
../datasets/scursulim_multiclase_p2_metadata.json.

Yahoo Finance / yfinance: auto_adjust=True (Close ajustado). Fechas de fin
INCLUSIVAS según el paper; en yfinance end es exclusiva y añadimos 1 día.
"""

import argparse
import hashlib
import json
from collections import OrderedDict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# El generador de escenarios ordena ALFABÉTICAMENTE asset_class. Prefijos
# explícitos permiten que los pesos [2,2,1,1,3] del escenario III se asignen
# conscientemente. Es una DECISIÓN PROPIA: el paper no da esta correspondencia.
CLASSES = OrderedDict([
    ("01_bonos_etf", ["AGG", "BND", "IEF", "TLT", "SHY"]),
    ("02_reits", ["PLD", "O", "AMT", "SPG", "EQIX"]),
    ("03_criptomonedas", ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "LTC-USD"]),
    ("04_materias_primas_etf", ["GLD", "SLV", "DBC", "DBA", "USO"]),
    ("05_acciones", ["AAPL", "MSFT", "AMZN", "JPM", "JNJ"]),
])
PERIODS = {
    "1": ("2023-12-18", "2024-12-16"),
    "2": ("2024-01-02", "2025-01-02"),
}


def extract_close(raw, tickers):
    """Compatible con la organización de MultiIndex de yfinance."""
    if raw is None or raw.empty:
        raise ValueError("Yahoo Finance no ha devuelto datos")
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw.xs("Close", axis=1, level=0)
        elif "Close" in raw.columns.get_level_values(-1):
            close = raw.xs("Close", axis=1, level=-1)
        else:
            raise ValueError(f"No aparece Close en las columnas: {raw.columns[:6]}")
    else:
        # Una sola columna sin multiíndice solo sería válida para un activo;
        # con nuestra lista fija de 25 constituye una respuesta malformada.
        raise ValueError("Descarga sin MultiIndex: se esperaban 25 tickers")
    missing = sorted(set(tickers) - set(close.columns))
    if missing:
        raise ValueError(f"No se han descargado estos tickers: {missing}")
    return close.loc[:, tickers].copy()


def main():
    parser = argparse.ArgumentParser(description="Prepara 25 activos de cinco clases financieras")
    parser.add_argument("--period", choices=PERIODS, default="2")
    parser.add_argument("--output-dir", type=Path, default=Path("../datasets"))
    args = parser.parse_args()

    try:
        import yfinance as yf
    except ImportError as ex:
        raise SystemExit("Falta yfinance. En el Pipenv del repositorio: pipenv install yfinance") from ex

    start, end_inclusive = PERIODS[args.period]
    end_exclusive = (date.fromisoformat(end_inclusive) + timedelta(days=1)).isoformat()
    tickers = [ticker for members in CLASSES.values() for ticker in members]
    if len(tickers) != 25 or len(set(tickers)) != 25:
        raise AssertionError("Los 25 símbolos deben ser distintos")

    prefix = f"scursulim_multiclase_p{args.period}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "prices": args.output_dir / f"{prefix}_prices.csv",
        "classes": args.output_dir / f"{prefix}_classes.csv",
        "metadata": args.output_dir / f"{prefix}_metadata.json",
    }
    already = [str(path) for path in paths.values() if path.exists()]
    if already:
        raise SystemExit(f"No se sobrescriben archivos existentes: {already}")

    print(f"Descargando {len(tickers)} activos desde {start} hasta {end_inclusive} (inclusive)...", flush=True)
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end_exclusive,
        auto_adjust=True,
        progress=False,
        threads=False,
        group_by="column",
    )
    close = extract_close(raw, tickers)
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.loc[start:end_inclusive].sort_index()
    close = close.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if close.index.duplicated().any():
        raise ValueError("Fechas duplicadas en la descarga")
    if (close <= 0).any().any():
        raise ValueError("Existen precios no positivos")

    missing_by_asset = close.isna().sum().to_dict()
    common = close.dropna(how="any")
    if len(common) < 180:
        raise ValueError(
            f"Solo quedan {len(common)} días comunes (se exigen al menos 180). "
            f"Nulos por activo: {missing_by_asset}"
        )
    if not common.index.is_monotonic_increasing:
        raise ValueError("Fechas desordenadas")
    if common.nunique().min() < 20:
        raise ValueError("Hay alguna serie de precios prácticamente constante")

    common.index.name = "Date"
    classes = pd.DataFrame(
        [(ticker, category) for category, members in CLASSES.items() for ticker in members],
        columns=["asset", "asset_class"],
    )
    # El generador debería seleccionar TODOS los activos, pues hay exactamente
    # cinco por clase; el orden dentro de cada bloque se fija con seed 42.
    common.to_csv(paths["prices"], float_format="%.12g", date_format="%Y-%m-%d")
    classes.to_csv(paths["classes"], index=False)

    def sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    metadata = {
        "dataset_type": "instancia_nueva_reproduccion_metodologica_NO_activos_originales",
        "paper_period": args.period,
        "requested_start_inclusive": start,
        "requested_end_inclusive": end_inclusive,
        "actual_first_date": str(common.index.min().date()),
        "actual_last_date": str(common.index.max().date()),
        "rows_before_intersection": int(len(close)),
        "rows_after_intersection": int(len(common)),
        "missing_by_ticker_before_intersection": {k: int(v) for k, v in missing_by_asset.items()},
        "source": "Yahoo Finance via yfinance",
        "yfinance_version": yf.__version__,
        "auto_adjust": True,
        "price_column": "Close ajustado",
        "alignment": "interseccion_de_fechas_sin_relleno_ni_interpolacion",
        "note": "ETFs usados como proxies de bonos y materias primas. Cinco clases definidas por nosotros; no identificadas en el paper.",
        "classes": CLASSES,
        "scenario_III_class_order_and_k": dict(zip(CLASSES, [2, 2, 1, 1, 3])),
        "prices_sha256": sha256(paths["prices"]),
        "classes_sha256": sha256(paths["classes"]),
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nDATASET CREADO (NO es la instancia numérica original del paper)")
    print("Filas completas:", len(common), "de", len(close))
    print("Primera fecha:", str(common.index.min().date()))
    print("Última fecha:", str(common.index.max().date()))
    print("Rangos de precios válidos:", bool((common > 0).all().all()))
    for class_name, members in CLASSES.items():
        print(f"  {class_name}: {', '.join(members)}")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
