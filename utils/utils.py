import pandas as pd
import numpy as np


def wap2vol(df: pd.DataFrame) -> float:
    """
    Compute realized volatility of WAP from log returns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with WAP prices.

    Returns
    -------
    float
        Realized volatility.
    """
    temp = np.log(df).diff()  # tick-to-tick log returns
    return np.sqrt(np.sum(temp**2))


def volGK(df: pd.DataFrame) -> float:
    """
    Compute Garman-Klass volatility estimator for a single time bucket.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'seconds_in_bucket' and 'WAP' columns.

    Returns
    -------
    float
        Garman-Klass volatility estimate.
    """
    # Pre-allocate for faster extraction
    sec_in_bucket = df["seconds_in_bucket"].values
    wap_values = df["WAP"].values

    open_tick = sec_in_bucket.argmin()
    close_tick = sec_in_bucket.argmax()
    high_tick = wap_values.argmax()
    low_tick = wap_values.argmin()

    open_price = wap_values[open_tick]
    close_price = wap_values[close_tick]
    high_price = wap_values[high_tick]
    low_price = wap_values[low_tick]

    log_hl = np.log(high_price / low_price)
    log_oc = np.log(open_price / close_price)

    vol = np.sqrt(0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_oc**2)
    return vol


def volGK_time_id(path: str) -> pd.Series:
    """
    Compute Garman-Klass volatility for each time_id in the order book.

    Parameters
    ----------
    path : str
        Path to parquet file containing order book data.

    Returns
    -------
    pd.Series
        Garman-Klass volatility for each time_id.
    """
    book = pd.read_parquet(path)

    # Calculate WAP
    p1, p2 = book["bid_price1"], book["ask_price1"]
    s1, s2 = book["bid_size1"], book["ask_size1"]
    book["WAP"] = (p1 * s2 + p2 * s1) / (s1 + s2)

    # Use groupby-apply to compute volatility
    return book.groupby("time_id")[["seconds_in_bucket", "WAP"]].apply(volGK)


def rel_vol_time_id(path: str) -> pd.Series:
    """
    Compute realized volatility (from WAP log returns) for each time_id.

    Parameters
    ----------
    path : str
        Path to parquet file containing order book data.

    Returns
    -------
    pd.Series
        Realized volatility for each time_id.
    """
    book = pd.read_parquet(path)

    # Calculate WAP
    p1, p2 = book["bid_price1"], book["ask_price1"]
    s1, s2 = book["bid_size1"], book["ask_size1"]
    book["WAP"] = (p1 * s2 + p2 * s1) / (s1 + s2)

    # Use groupby-agg to compute realized volatility
    return book.groupby("time_id")["WAP"].agg(wap2vol)


def wap(path: str) -> pd.DataFrame:
    """
    Compute WAP for each row of order book data.

    Parameters
    ----------
    path : str
        Path to parquet file containing order book data.

    Returns
    -------
    pd.DataFrame
        Order book DataFrame with WAP column added.
    """
    book = pd.read_parquet(path)

    # Calculate WAP
    p1, p2 = book["bid_price1"], book["ask_price1"]
    s1, s2 = book["bid_size1"], book["ask_size1"]
    book["WAP"] = (p1 * s2 + p2 * s1) / (s1 + s2)

    return book


def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))
