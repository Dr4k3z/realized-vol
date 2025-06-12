import numpy as np
import pandas as pd
import glob
import re
import contextlib
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed
from .utils import rel_vol_time_id, volGK_time_id

import warnings

warnings.filterwarnings("ignore")


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


def volume_imbalance_time_id(path: str) -> pd.Series:
    """Computes the average volume imbalance per time_id."""
    book = pd.read_parquet(path)
    book["volume_imbalance"] = (book["ask_size1"] - book["bid_size1"]) / 100
    return book.groupby("time_id")["volume_imbalance"].mean()


def bid_ask_spread_time_id(path: str) -> pd.Series:
    """Computes the average bid-ask spread per time_id."""
    book = pd.read_parquet(path)
    book["bid_ask_spread"] = book["ask_price1"] - book["bid_price1"]
    return book.groupby("time_id")["bid_ask_spread"].mean()


class LinearRegressionETL:
    """
    ETL class to load, process, and join feature data for linear regression modeling
    on order book and realized volatility datasets.
    """

    def __init__(self, orderbook_path: Path, train_path: Path):
        """
        Initializes the ETL class with paths to data.

        Parameters:
            orderbook_path (Path): Directory containing the parquet order book files.
            train_path (Path): CSV file path to training dataset.
        """
        self.orderbook_path = orderbook_path
        self.train_path = train_path

        self.train: pd.DataFrame = pd.read_csv(self.train_path)
        self.orderbook: list[str] = glob.glob(self.orderbook_path)

    @property
    def train_df(self) -> pd.DataFrame:
        """Returns the loaded training DataFrame."""
        return self.train

    def denormalize_prices(self) -> pd.DataFrame:
        """Parallelized computation of denormalized prices for each orderbook file with tqdm."""
        with tqdm(
            tqdm(desc="Denormalizing prices", total=len(self.orderbook), unit="file")
        ):
            df_prices_denorm = pd.concat(
                Parallel(n_jobs=-1)(
                    delayed(calc_prices)(file_path) for file_path in self.orderbook
                )
            )

        self._denormalized_prices = df_prices_denorm
        return df_prices_denorm

    @property
    def denormalized_prices(self) -> pd.DataFrame:
        """Returns cached denormalized prices, computing them if necessary."""
        if not hasattr(self, "_denormalized_prices"):
            self._denormalized_prices = self.denormalize_prices()
        return self._denormalized_prices

    def compute_features(self) -> pd.DataFrame:
        """
        Computes features including realized volatility, volume imbalance,
        bid-ask spread, and joins them with training and denormalized price data.

        Returns:
            pd.DataFrame: The final joined and cleaned dataset.
        """
        stock_id = []
        time_id = []
        relvol = []
        volgk = []
        volume_imbalance = []
        bid_ask_spread = []

        if not hasattr(self, "_denormalized_prices"):
            self.denormalize_prices()

        with tqdm(
            total=len(self.orderbook), desc="Computing features", unit="file"
        ) as pbar:
            for file_path in self.orderbook:
                sid = regex_stock_id(file_path)
                temp_relvol = rel_vol_time_id(file_path)
                temp_volgk = volGK_time_id(file_path)
                temp_imbalance = volume_imbalance_time_id(file_path)
                temp_bidask = bid_ask_spread_time_id(file_path)

                n = len(temp_relvol)
                stock_id.extend([sid] * n)
                time_id.extend(temp_relvol.index)
                relvol.extend(temp_relvol.values)
                volgk.extend(temp_volgk.values)
                volume_imbalance.extend(temp_imbalance.values)
                bid_ask_spread.extend(temp_bidask.values)
                pbar.update(1)

        past_vol = pd.DataFrame(
            {
                "stock_id": stock_id,
                "time_id": time_id,
                "rel_vol": relvol,
                "vol_gk": volgk,
                "imbalance": volume_imbalance,
                "bidask": bid_ask_spread,
            }
        )

        prices_df = self.denormalized_prices
        past_vol = past_vol.merge(prices_df, on=["stock_id", "time_id"], how="left")
        joined = self.train.merge(past_vol, on=["stock_id", "time_id"])
        joined.dropna(inplace=True)
        joined.set_index(["stock_id"], inplace=True)

        return joined
