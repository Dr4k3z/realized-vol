import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
from glob import glob
from pathlib import Path
from joblib import Parallel, delayed
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
from .utils import calc_prices, calc_rv

warnings.filterwarnings("ignore")


def plot_emb(emb, color, name, kind="volatility", fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
        plt.grid(True)
    if kind == "volatility":
        norm = mpl.colors.LogNorm()
        ticks = mpl.ticker.LogLocator(2)
        formatter = mpl.ticker.ScalarFormatter()
    elif kind == "date":
        norm = None
        ticks = None
        formatter = mpl.dates.AutoDateFormatter(mpl.dates.MonthLocator())
    plot = ax.scatter(
        emb[:, 0], emb[:, 1], s=3, c=color, edgecolors="none", cmap="jet", norm=norm
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cb = fig.colorbar(plot, label=kind, format=formatter, ticks=ticks, cax=cax)
    cb.ax.minorticks_off()
    ax.set_title(f"{name}")


class TemporalOrderETL:
    def __init__(self, orderbook_path: Path, train_path: Path):
        self.orderbook_path = orderbook_path
        self.train_path = train_path

        self.train: pd.DataFrame = pd.read_csv(self.train_path)
        self.orderbook: list[str] = glob(self.orderbook_path)

    @classmethod
    def load(cls, orderbook_path: Path, train_path: Path):
        etl = cls(orderbook_path, train_path)
        data_dir = "optiver-realized-volatility-prediction"

        etl._df_files = (
            pd.DataFrame(
                {"book_path": glob(f"{data_dir}/book_train.parquet/**/*.parquet")}
            )
            .assign(
                stock_id=lambda x: x.book_path.str.extract("stock_id=(\d+)").astype("int")
            )
            .sort_values("stock_id")
        )
        etl._df_target_train = pd.read_csv(f"{data_dir}/train.csv")
        etl._df_volatility_train = etl._df_target_train.groupby("time_id").target.mean()

        return etl

    @property
    def df_files(self):
        if not hasattr(self, "_df_files"):
            self._df_files = (
                pd.DataFrame(
                    {
                        "book_path": glob(
                            f"{self.orderbook_path}/book_train.parquet/**/*.parquet"
                        )
                    }
                )
                .assign(
                    stock_id=lambda x: x.book_path.str.extract("stock_id=(\d+)").astype(
                        "int"
                    )
                )
                .sort_values("stock_id")
            )
        return self._df_files

    @property
    def df_target_train(self):
        if not hasattr(self, "_df_target_train"):
            self._df_target_train = pd.read_csv(self.train_path / "train.csv")
        return self._df_target_train

    @property
    def df_volatility_train(self):
        if not hasattr(self, "_df_volatility_train"):
            self._df_volatility_train = self.df_target_train.groupby(
                "time_id"
            ).target.mean()
        return self._df_volatility_train

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
        return df_prices_denorm.pivot(index="time_id", columns="stock_id", values="price")

    @property
    def denormalized_prices(self) -> pd.DataFrame:
        """Returns cached denormalized prices, computing them if necessary."""
        if not hasattr(self, "_denormalized_prices"):
            self._denormalized_prices = self.denormalize_prices()
        return self._denormalized_prices

    @property
    def df_rv_train(self) -> pd.DataFrame:
        """Returns the realized volatility for the training set."""
        df_rv_train = pd.concat(
            Parallel(n_jobs=-1, verbose=0)(
                delayed(calc_rv)(r) for _, r in self._df_files.iterrows()
            )
        )
        return df_rv_train
