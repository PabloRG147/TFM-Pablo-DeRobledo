import numpy as np


def calculate_portfolio_return(
    x: np.ndarray, weights: np.ndarray, assets_returns: np.ndarray
) -> float:
    """Computes portfolio return.
    
    Args:
        x (np.ndarray): A binary array with the assets that were
        selected to compose the portfolio.
        weights (np.ndarray): An array with assets weights.
        assets_returns (np.ndarray): An array with assets returns.

    Returns: Portfolio return.
    """

    return assets_returns @ (x * weights)


def calculate_portfolio_risk(
    x: np.ndarray, weights: np.ndarray, covariance_matrix: np.ndarray
) -> float:
    """Computes portfolio risk.
    
    Args:
        x (np.ndarray): A binary array with the assets that were
        selected to compose the portfolio.
        weights (np.ndarray): An array with assets weights.
        covariance_matrix (np.ndarray): A matrix with the covariance
        between assets.

    Returns: Portfolio risk.
    """

    return np.sqrt((x * weights).T @ covariance_matrix @ (x * weights))


def calculate_portfolio_sharpe_ratio(
    x: np.ndarray,
    weights: np.ndarray,
    assets_returns: np.ndarray,
    covariance_matrix: np.ndarray,
    return_risk_free: float = 0.0375,
) -> float:
    """Computes portfolio Sharpe ratio.
    
    Args:
        x (np.ndarray): A binary array with the assets that were
        selected to compose the portfolio.
        weights (np.ndarray): An array with assets weights.
        assets_returns (np.ndarray): An array with assets returns.
        covariance_matrix (np.ndarray): A matrix with the covariance
        between assets.
        return_risk_free (float, optional): The return risk free rate. Defaults to 0.0375.

    Returns: Portfolio Sharpe ratio.
    """

    portfolio_return = calculate_portfolio_return(
        x=x, weights=weights, assets_returns=assets_returns
    )
    portfolio_risk = calculate_portfolio_risk(
        x=x, weights=weights, covariance_matrix=covariance_matrix
    )

    return (portfolio_return - return_risk_free) / portfolio_risk
