from pathlib import Path
import pandas as pd

folder = Path(__file__).resolve().parent
histories = folder / "scenarioI" / "results" / "histories"

for file in sorted(histories.glob("*shots256_budget30*.pkl")):
    df = pd.read_pickle(file)

    print(f"\nArchivo: {file.name}")
    print(f"Evaluaciones: {len(df)}")
    print(f"Columnas: {list(df.columns)}")
    print(df[["evaluation", "energy"]].head())

    assert not df.empty, "Historial vacío"
    assert df["energy"].notna().all(), "Hay energías no válidas"
