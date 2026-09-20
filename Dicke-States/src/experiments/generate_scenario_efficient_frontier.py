import sys

sys.path.append("../")

import argparse
import json

import gurobipy as gp
import numpy as np
import pandas as pd

from gurobipy import GRB
from tqdm import tqdm

from utils import (
    calculate_portfolio_return,
    calculate_portfolio_risk,
    calculate_portfolio_sharpe_ratio,
)

parser = argparse.ArgumentParser(description="Inputs to generate experiments scenarios")
parser.add_argument(
    "--scenario",
    choices=["scenarioI", "scenarioII", "scenarioIII"],
    help="Experiment scenario",
    required=True,
)
parser.add_argument("--seed", type=int, help="The seed number", default=42)
args = parser.parse_args()

SCENARIO = args.scenario
SEED = args.seed

return_risk_free = 0.0375
np.random.seed(seed=SEED)
assets_data = pd.read_csv(
    filepath_or_buffer="../datasets/sp500_assets_close_price.csv", index_col=0
)

columns = [
    "aversion_risk",
    "obj_func",
    "answer",
    "portfolio_return",
    "portfolio_risk",
    "sharpe",
    "is_feasible",
]
data = []
env = gp.Env(params={"OutputFlag": 0})

if SCENARIO == "scenarioI":

    tickers = np.random.choice(assets_data.columns, size=11, replace=False)
    assets_close_price = assets_data[tickers]
    assets_pct_change = assets_close_price.pct_change().dropna()
    covariance_annualized = assets_pct_change.cov() * np.sqrt(252)
    returns_annualized = assets_pct_change.mean() * 252
    weights_array = np.ones(shape=(len(tickers),))

    k = 4

    for q in tqdm(np.linspace(0, 1, 101)):
        model = gp.Model("portfolio", env=env)
        x = np.array(
            [
                model.addVar(name=f"x({i})", vtype=GRB.BINARY)
                for i in range(assets_close_price.columns.shape[0])
            ]
        )
        model.setObjective(
            q
            * (
                (x * weights_array).T
                @ covariance_annualized.values
                @ (x * weights_array)
            )
            - (1 - q) * (returns_annualized.values @ ((x * weights_array)))
            - return_risk_free
        )
        model.addConstr(x.sum() <= k)
        model.optimize()
        result_array = np.array(
            [
                model.getVarByName(f"x({i})").X
                for i in range(assets_close_price.columns.shape[0])
            ]
        ).astype(int)
        data.append(
            [
                q,
                model.ObjVal,
                result_array,
                calculate_portfolio_return(
                    x=result_array,
                    weights=weights_array,
                    assets_returns=returns_annualized.values,
                ),
                calculate_portfolio_risk(
                    x=result_array,
                    weights=weights_array,
                    covariance_matrix=covariance_annualized.values,
                ),
                calculate_portfolio_sharpe_ratio(
                    x=result_array,
                    weights=weights_array,
                    assets_returns=returns_annualized.values,
                    covariance_matrix=covariance_annualized.values,
                    return_risk_free=return_risk_free,
                ),
                bool(result_array.sum() <= k),
            ]
        )
    df_efficient_frontier = pd.DataFrame(data=data, columns=columns)
    df_efficient_frontier.to_pickle(f"{SCENARIO}/df_efficient_frontier.pkl")
elif SCENARIO == "scenarioII":

    tickers = np.random.choice(assets_data.columns, size=11, replace=False)
    assets_close_price = assets_data[tickers]
    assets_pct_change = assets_close_price.pct_change().dropna()
    covariance_annualized = assets_pct_change.cov() * np.sqrt(252)
    returns_annualized = assets_pct_change.mean() * 252
    weights_array = np.ones(shape=(len(tickers),))

    k_min = 3
    k_max = 6

    for q in tqdm(np.linspace(0, 1, 101)):
        model = gp.Model("portfolio", env=env)
        x = np.array(
            [
                model.addVar(name=f"x({i})", vtype=GRB.BINARY)
                for i in range(assets_close_price.columns.shape[0])
            ]
        )
        model.setObjective(
            q
            * (
                (x * weights_array).T
                @ covariance_annualized.values
                @ (x * weights_array)
            )
            - (1 - q) * (returns_annualized.values @ ((x * weights_array)))
            - return_risk_free
        )
        model.addConstr(x.sum() <= k_max)
        model.addConstr(x.sum() >= k_min)
        model.optimize()
        result_array = np.array(
            [
                model.getVarByName(f"x({i})").X
                for i in range(assets_close_price.columns.shape[0])
            ]
        ).astype(int)
        data.append(
            [
                q,
                model.ObjVal,
                result_array,
                calculate_portfolio_return(
                    x=result_array,
                    weights=weights_array,
                    assets_returns=returns_annualized.values,
                ),
                calculate_portfolio_risk(
                    x=result_array,
                    weights=weights_array,
                    covariance_matrix=covariance_annualized.values,
                ),
                calculate_portfolio_sharpe_ratio(
                    x=result_array,
                    weights=weights_array,
                    assets_returns=returns_annualized.values,
                    covariance_matrix=covariance_annualized.values,
                    return_risk_free=return_risk_free,
                ),
                bool(result_array.sum() <= k_max) and bool(result_array.sum() >= k_min),
            ]
        )
    df_efficient_frontier = pd.DataFrame(data=data, columns=columns)
    df_efficient_frontier.to_pickle(f"{SCENARIO}/df_efficient_frontier.pkl")
elif SCENARIO == "scenarioIII":
    with open(file="../datasets/tickers_by_sectors.json", mode="rt") as f:
        tickers_by_sectors = json.load(f)

    np.random.seed(seed=42)
    assets_classes = np.random.choice(list(tickers_by_sectors.keys()), size=4)
    assets_by_class = {
        value: list(np.random.choice(tickers_by_sectors[value], size=5))
        for value in assets_classes
    }

    constraints_dict = {
        "Energy": ("eq", 3),
        "Financial Services": ("gleq", 1, 2),
        "Real Estate": ("eq", 2),
        "Basic Materials": ("leq", 4),
    }

    tickers = []
    for value in assets_by_class.values():
        tickers.extend(list(value))
    assets_close_price = assets_data[tickers]

    assets_pct_change = assets_close_price.pct_change().dropna()
    covariance_annualized = assets_pct_change.cov() * np.sqrt(252)
    returns_annualized = assets_pct_change.mean() * 252
    weights_array = np.ones(shape=(len(tickers),))

    constraints_matrix = np.zeros(shape=(len(assets_by_class.keys()), len(tickers)))

    for idx, key in enumerate(assets_by_class.keys()):
        constraints_matrix[idx, :] = assets_close_price.columns.isin(
            assets_by_class[key]
        ).astype(int)

    for q in tqdm(np.linspace(0, 1, 101)):
        model = gp.Model("portfolio", env=env)
        x = np.array(
            [
                model.addVar(name=f"x({i})", vtype=GRB.BINARY)
                for i in range(assets_close_price.columns.shape[0])
            ]
        )
        model.setObjective(
            q
            * (
                (x * weights_array).T
                @ covariance_annualized.values
                @ (x * weights_array)
            )
            - (1 - q) * (returns_annualized.values @ ((x * weights_array)))
            - return_risk_free
        )
        for idx, key in enumerate(assets_by_class.keys()):
            tp = constraints_dict[key]
            if tp[0] == "eq":
                model.addConstr(constraints_matrix[idx] @ x == tp[1])
            elif tp[0] == "leq":
                model.addConstr(constraints_matrix[idx] @ x <= tp[1])
            elif tp[0] == "geq":
                model.addConstr(constraints_matrix[idx] @ x >= tp[1])
            elif tp[0] == "gleq":
                model.addConstr(constraints_matrix[idx] @ x >= tp[1])
                model.addConstr(constraints_matrix[idx] @ x <= tp[2])
        model.optimize()
        result_array = np.array(
            [
                model.getVarByName(f"x({i})").X
                for i in range(assets_close_price.columns.shape[0])
            ]
        ).astype(int)
        check1 = (constraints_matrix[0] @ result_array) == 3
        check2 = (constraints_matrix[1] @ result_array) >= 1 and (
            constraints_matrix[1] @ result_array
        ) <= 2
        check3 = (constraints_matrix[2] @ result_array) == 2
        check4 = (constraints_matrix[3] @ result_array) <= 4
        data.append(
            [
                q,
                model.ObjVal,
                result_array,
                calculate_portfolio_return(
                    x=result_array,
                    weights=weights_array,
                    assets_returns=returns_annualized.values,
                ),
                calculate_portfolio_risk(
                    x=result_array,
                    weights=weights_array,
                    covariance_matrix=covariance_annualized.values,
                ),
                calculate_portfolio_sharpe_ratio(
                    x=result_array,
                    weights=weights_array,
                    assets_returns=returns_annualized.values,
                    covariance_matrix=covariance_annualized.values,
                    return_risk_free=return_risk_free,
                ),
                np.allclose([check1, check2, check3, check4], [True for _ in range(4)]),
            ]
        )
    df_efficient_frontier = pd.DataFrame(data=data, columns=columns)
    df_efficient_frontier.to_pickle(f"{SCENARIO}/df_efficient_frontier.pkl")
