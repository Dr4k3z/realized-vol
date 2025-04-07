import pandas as pd
import numpy as np


def wap(book: pd.DataFrame) -> pd.Series:
    """
    WAP: Weighted Average Price
    """
    totalSize = book["bid_size1"] + book["ask_size1"]
    return (
        book["bid_price1"] * book["ask_size1"] + book["ask_price1"] * book["bid_size1"]
    ) / totalSize


def log_return(prices: pd.Series) -> pd.Series:
    """
    Log return
    """
    return np.log(prices).diff()


def realized_volatility(series_log_return: pd.Series | np.ndarray) -> np.ndarray:
    """
    Realized volatility
    """
    return np.sqrt(np.sum(series_log_return**2))


def wap2vol(wap: np.ndarray) -> np.ndarray:
    """
    WAP to volatility
    """
    # Calculate the log returns
    log_returns = log_return(wap)

    # Calculate the realized volatility
    return realized_volatility(log_returns)
