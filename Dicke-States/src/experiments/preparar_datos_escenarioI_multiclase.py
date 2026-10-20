#!/usr/bin/env python3
"""Prepara 10 acciones (escenario I) con el MISMO calendario que el dataset
multiclase p2. No altera archivos existentes ni crea escenarios VQE.

Uso desde src/experiments:
    pipenv run python preparar_datos_escenarioI_real.py

Conserva del dataset p2 cinco acciones (AAPL, MSFT, AMZN, JPM, JNJ) y descarga
cinco adicionales de Yahoo Finance (GOOGL, NVDA, XOM, PG, WMT) con
`auto_adjust=True`. Se exige cobertura COMPLETA en las fechas del dataset p2:
no se rellenan, interpolan ni eliminan fechas. Así, las observaciones y el
periodo de rendimientos son idénticos a los usados en II/III.

La selección de activos es una instancia NUEVA: NO se afirma que sean los del
artículo de Scursulim ni que reproduzcan sus resultados numéricos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

EXISTING_STOCKS = ["AAPL", "MSFT", "AMZN", "JPM", "JNJ"]
ADDED_STOCKS = ["GOOGL", "NVDA", "XOM", "PG", "WMT"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_adjusted_close(result: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if result is None or result.empty:
        raise ValueError("Yahoo Finance no ha devuelto datos")
    if not isinstance(result.columns, pd.MultiIndex):
        raise ValueError("La descarga de los cinco tickers no tiene columnas MultiIndex")
    if "Close" in result.columns.get_level_values(0):
        close = result.xs("Close", axis=1, level=0)
    elif "Close" in result.columns.get_level_values(-1):
        close = result.xs("Close", axis=1, level=-1)
    else:
        raise ValueError("Falta el campo Close en los datos descargados")
    close.columns = close.columns.map(str)
    missing = sorted(set(tickers) - set(close.columns))
    if missing:
        raise ValueError(f"Faltan tickers en la descarga: {missing}")
    return close.loc[:, tickers].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--multiclass-prices", type=Path,
        default=Path("../datasets/scursulim_multiclase_p2_prices.csv"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("../datasets/scursulim_scenarioI_p2_prices.csv"),
    )
    args = parser.parse_args()
    if not args.multiclass_prices.is_file():
        parser.error(f"Falta {args.multiclass_prices}")
    metadata_file = args.output.with_name(args.output.stem.replace("_prices", "") + "_metadata.json")
    if args.output.exists() or metadata_file.exists():
        parser.error(f"No se sobrescriben archivos existentes: {args.output}, {metadata_file}")

    # Los 25 activos del escenario II/III ya están alineados por fechas.
    base = pd.read_csv(args.multiclass_prices, index_col=0)
    base.index = pd.to_datetime(base.index, errors="raise").normalize()
    base = base.sort_index()
    if base.index.duplicated().any() or not base.index.is_monotonic_increasing:
        raise ValueError("Índice de fechas inválido en el CSV multiclase")
    if len(base) < 180:
        raise ValueError(f"Muy pocas fechas comunes: {len(base)}")
    missing = sorted(set(EXISTING_STOCKS) - set(base.columns))
    if missing:
        raise ValueError(f"El CSV multiclase no tiene: {missing}")
    existing = base[EXISTING_STOCKS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(existing.to_numpy()).all() or (existing <= 0).any().any():
        raise ValueError("Las cinco acciones existentes presentan precios inválidos")

    try:
        import yfinance as yf
    except ImportError as ex:
        raise SystemExit("Falta yfinance; instalar en Pipenv: pipenv install yfinance") from ex

    first_day = base.index.min().date()
    last_day = base.index.max().date()
    end_exclusive = (last_day + timedelta(days=1)).isoformat()
    print(
        f"Descargando 5 acciones adicionales entre {first_day} y {last_day} (inclusive)...",
        flush=True,
    )
    raw = yf.download(
        tickers=ADDED_STOCKS,
        start=first_day.isoformat(),
        end=end_exclusive,
        auto_adjust=True,
        progress=False,
        threads=False,
        group_by="column",
    )
    added = extract_adjusted_close(raw, ADDED_STOCKS)
    idx = pd.to_datetime(added.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    added.index = idx.normalize()
    if added.index.duplicated().any():
        raise ValueError("Yahoo Finance devolvió fechas duplicadas")
    added = added.apply(pd.to_numeric, errors="coerce").reindex(base.index)

    # IMPORTANTE: exigir las mismas fechas que II/III, sin cambiar retornos.
    bad = ~np.isfinite(added.to_numpy()) | (added.to_numpy() <= 0)
    if bad.any():
        row, col = np.argwhere(bad)[0]
        raise ValueError(
            "Precios faltantes o inválidos en las fechas ya usadas por II/III: "
            f"primero {ADDED_STOCKS[col]} el {base.index[row].date()}. "
            "No se modifica el calendario automáticamente."
        )

    all_prices = pd.concat([existing, added], axis=1)
    assert all_prices.shape == (len(base), 10)
    assert all_prices.index.equals(base.index)
    assert all_prices.columns.is_unique
    all_prices.index.name = "Date"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_prices.to_csv(args.output, float_format="%.12g", date_format="%Y-%m-%d")
    metadata = {
        "type": "escenario_I_10_acciones_nueva_instancia_no_numeros_paper",
        "source": "Yahoo Finance via yfinance; cinco precios heredados del CSV multiclase",
        "yfinance_version": yf.__version__,
        "existing_stocks_from_multiclass": EXISTING_STOCKS,
        "extra_stocks_downloaded": ADDED_STOCKS,
        "stocks_in_csv": list(all_prices.columns),
        "auto_adjust": True,
        "start_inclusive": first_day.isoformat(),
        "end_inclusive": last_day.isoformat(),
        "common_price_rows": int(len(all_prices)),
        "return_rows": int(len(all_prices) - 1),
        "calendar_matches_multiclass": bool(all_prices.index.equals(base.index)),
        "missing_filling": "ninguno",
        "multiclass_prices_path": str(args.multiclass_prices.resolve()),
        "multiclass_prices_sha256": sha256(args.multiclass_prices),
        "scenarioI_prices_sha256": sha256(args.output),
    }
    metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nDATASET DEL ESCENARIO I CREADO")
    print("Acciones:", ", ".join(all_prices.columns))
    print("Fechas:", first_day, "a", last_day)
    print("Filas de precios:", len(all_prices), "| filas de rendimientos:", len(all_prices)-1)
    print("Mismo calendario que escenarios II/III:", metadata["calendar_matches_multiclass"])
    print("Precios:", args.output)
    print("Metadatos:", metadata_file)
    print("NOTA: instancia propia para reproducir la metodología, no los números del paper.")


if __name__ == "__main__":
    main()
