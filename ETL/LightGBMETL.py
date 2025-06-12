import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import warnings
from pathlib import Path
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", 300)
pd.set_option("display.max_columns", 300)


def calc_wap(df):
    wap = (df["bid_price1"] * df["ask_size1"] + df["ask_price1"] * df["bid_size1"]) / (
        df["bid_size1"] + df["ask_size1"]
    )
    return wap


def calc_wap2(df):
    wap = (df["bid_price2"] * df["ask_size2"] + df["ask_price2"] * df["bid_size2"]) / (
        df["bid_size2"] + df["ask_size2"]
    )
    return wap


def log_return(list_stock_prices):
    return np.log(list_stock_prices).diff()


def realized_volatility(series):
    return np.sqrt(np.sum(series**2))


def count_unique(series):
    return len(np.unique(series))


class LightGBMETL:
    def __init__(self, orderbook_path: Path, train_path: Path):
        self.orderbook_path = orderbook_path
        self.train_path = train_path

        self._train = pd.read_csv(self.train_path)
        self.list_stock_ids = self._train["stock_id"].unique()

        self.data_dir = "optiver-realized-volatility-prediction/"

    @property
    def train(self) -> pd.DataFrame:
        return self._train

    def _preprocessor_book(self, file_path):
        df = pd.read_parquet(file_path)

        # calculate return etc
        df["wap"] = calc_wap(df)
        df["log_return"] = df.groupby("time_id")["wap"].transform(log_return)

        df["wap2"] = calc_wap2(df)
        df["log_return2"] = df.groupby("time_id")["wap2"].transform(log_return)

        df["wap_balance"] = abs(df["wap"] - df["wap2"])

        df["price_spread"] = (df["ask_price1"] - df["bid_price1"]) / (
            (df["ask_price1"] + df["bid_price1"]) / 2
        )
        df["bid_spread"] = df["bid_price1"] - df["bid_price2"]
        df["ask_spread"] = df["ask_price1"] - df["ask_price2"]
        df["total_volume"] = (df["ask_size1"] + df["ask_size2"]) + (
            df["bid_size1"] + df["bid_size2"]
        )
        df["volume_imbalance"] = abs(
            (df["ask_size1"] + df["ask_size2"]) - (df["bid_size1"] + df["bid_size2"])
        )

        # dict for aggregate
        create_feature_dict = {
            "log_return": [realized_volatility],
            "log_return2": [realized_volatility],
            "wap_balance": [np.mean],
            "price_spread": [np.mean],
            "bid_spread": [np.mean],
            "ask_spread": [np.mean],
            "volume_imbalance": [np.mean],
            "total_volume": [np.mean],
            "wap": [np.mean],
        }

        #####groupby / all seconds
        df_feature = pd.DataFrame(
            df.groupby(["time_id"]).agg(create_feature_dict)
        ).reset_index()

        df_feature.columns = [
            "_".join(col) for col in df_feature.columns
        ]  # time_id is changed to time_id_

        ######groupby / last XX seconds
        last_seconds = [300]

        for second in last_seconds:
            second = 600 - second

            df_feature_sec = pd.DataFrame(
                df.query(f"seconds_in_bucket >= {second}")
                .groupby(["time_id"])
                .agg(create_feature_dict)
            ).reset_index()

            df_feature_sec.columns = [
                "_".join(col) for col in df_feature_sec.columns
            ]  # time_id is changed to time_id_

            df_feature_sec = df_feature_sec.add_suffix("_" + str(second))

            df_feature = pd.merge(
                df_feature,
                df_feature_sec,
                how="left",
                left_on="time_id_",
                right_on=f"time_id__{second}",
            )
            df_feature = df_feature.drop([f"time_id__{second}"], axis=1)

        # create row_id
        stock_id = file_path.split("=")[1]
        df_feature["row_id"] = df_feature["time_id_"].apply(lambda x: f"{stock_id}-{x}")
        df_feature = df_feature.drop(["time_id_"], axis=1)

        return df_feature

    def preprocessor(self) -> pd.DataFrame:
        # parallel computing to save time

        df = pd.DataFrame()

        def for_joblib(stock_id):
            file_path_book = (
                self.data_dir + "book_train.parquet/stock_id=" + str(stock_id)
            )
            df_tmp = self._preprocessor_book(file_path_book)

            return df_tmp

        df = Parallel(n_jobs=-1, verbose=1)(
            delayed(for_joblib)(stock_id) for stock_id in self.list_stock_ids
        )

        df = pd.concat(df, ignore_index=True)
        return df
