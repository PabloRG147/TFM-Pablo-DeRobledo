# Imports
import json
import os
import pathlib

import pandas as pd

# Creating a list with files path.
files_list = [
    str(path)
    for path in pathlib.Path("../datasets/").glob("assets_*.csv")
    if path.is_file()
]

# Load the files in the list above.
assets_data_list = [
    pd.read_csv(filepath_or_buffer=f"{file_path}").dropna(axis=1)
    for file_path in files_list
]

# Concatenation of the DataFrames.
for idx, df in enumerate(assets_data_list):
    df = df.set_index("Date")
    if idx == 0:
        df_aux = df.copy()
    else:
        df_aux = pd.concat([df_aux, df], axis=1)

# Deleting files
for file_path in files_list:
    if os.path.exists(file_path):
        os.remove(file_path)

# Assets anonimization
df_sp = pd.read_csv(filepath_or_buffer="../datasets/sp500_companies.csv")
mask = df_sp.Symbol.isin(df_aux.columns)
df_sp = df_sp[mask]
new_columns = {col: f"asset_{idx}" for idx, col in enumerate(df_aux.columns)}
df_sp.Symbol = df_sp.Symbol.map(new_columns)
df_aux.columns = df_aux.columns.map(new_columns)
df_aux.to_csv("../datasets/sp500_assets_close_price.csv")

# Creating a dictionary with the assets by sectors.
tickers_by_sectors = {
    sector: df_sp[df_sp.Sector == sector].Symbol.values.tolist()
    for sector in df_sp.Sector.unique()
}

# Saving the dictionary as a JSON file.
with open(file="../datasets/tickers_by_sectors.json", mode="wt") as f:
    json.dump(tickers_by_sectors, f)
