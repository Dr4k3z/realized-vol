import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re


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


def calc_model_importance(model, feature_names=None, importance_type="gain"):
    importance_df = pd.DataFrame(
        model.feature_importance(importance_type=importance_type),
        index=feature_names,
        columns=["importance"],
    ).sort_values("importance")
    return importance_df


def calc_mean_importance(importance_df_list):
    mean_importance = np.mean(
        np.array([df["importance"].values for df in importance_df_list]), axis=0
    )
    mean_df = importance_df_list[0].copy()
    mean_df["importance"] = mean_importance
    return mean_df


def plot_importance(importance_df, title="", save_filepath=None, figsize=(8, 12)):
    fig, ax = plt.subplots(figsize=figsize)
    importance_df.plot.barh(ax=ax)
    if title:
        plt.title(title)
    plt.tight_layout()
    if save_filepath is None:
        plt.show()
    else:
        plt.savefig(save_filepath)
    plt.close()


def regex_stock_id(file_path: str) -> int:
    """Extracts the stock_id from a given file path using regex."""
    match = re.search(r"stock_id=(\d+)", file_path)
    return int(match.group(1)) if match else -1


def calc_price(df: pd.DataFrame) -> float:
    """Estimates the price scale from bid/ask spreads for a single time_id group."""
    diff = abs(df.diff())
    min_diff = np.nanmin(diff.where(lambda x: x > 0))
    if pd.isna(min_diff) or min_diff == 0:
        return np.nan
    n_ticks = (diff / min_diff).round()
    return 0.01 / np.nanmean(diff / n_ticks)


def calc_prices(file_path: str) -> pd.DataFrame:
    """Calculates denormalized prices for each time_id in a given file."""
    df = pd.read_parquet(
        file_path,
        columns=["time_id", "ask_price1", "ask_price2", "bid_price1", "bid_price2"],
    )
    price_df = (
        df.groupby("time_id", group_keys=False)
        .apply(calc_price)
        .to_frame("price")
        .reset_index()
    )
    price_df["stock_id"] = regex_stock_id(file_path)
    return price_df


def calc_rv(r):
    df = pd.read_parquet(r.book_path)
    df["wap"] = (df.ask_price1 * df.bid_size1 + df.bid_price1 * df.ask_size1) / (
        df.ask_size1 + df.bid_size1
    )
    df = (
        df.groupby("time_id")
        .wap.apply(lambda x: (np.log(x).diff() ** 2).sum() ** 0.5)
        .reset_index()
    )
    df.rename(columns={"wap": "rv"}, inplace=True)
    df["stock_id"] = r.stock_id
    return df
