# Forecasting Realized Volatility: an empirical study

![GitHub last commit (branch)](https://img.shields.io/github/last-commit/Dr4k3z/realized-vol/main)

This work has been developed by [Matteo Campagnoli](@Dr4k3z), [Giacomo Kirn](@jackirn), [Riccardo Girgenti](@rikygirg) and [Francesco Ligorio](@lygobot), during the Econometrics course at PoliMi, Academic Yaer 2024-2025.

Starting from the dataset provided by Optiver in the Kaggle "Forecasting Realized Volatility Challege", we explored various techniques to predict realized volatility. 

In the repo, you'll find three notebooks, each implementing a different approach:
    
    1. `ols.ipynb` Linear Regression

    2. `lgbm.ipynb` Light Gradient Boosting Machine

    3. `temporal-order.ipynb` Temporal Order reconstruction

For more details, please read the extensive [project report](Forecasting_Realized_Vol.pdf). A final warning: the dataset is huge and to run our algorithms fastly we largely used Python `Parallel` package, which support the parallerization of processes on the CPU. We always take the maximum number of core available, `Parallel(n_jobs=-1)`; this is the fastest solution, but not necessarily the safest. Be aware of your computer resources before running this.

## Installation

To run the code, execute the command

```pip install -r requirements.txt```

to install all the relevant python packages. Some of these may require a specific version of `numpy` or `sklearn`, hence we suggest to install them in a new clean Python enviroment. To rapidly set up the virtual enviroment, we suggest to use this tool: [uv](https://docs.astral.sh/uv/).

## Input and Output

The code needs the Optiver dataset, which, due to his size, has not been included in the repo. By default, the notebooks look for the dataset in the ```./optiver-realized-volatility-prediction/``` folder, but you should be able to easily change that just by modifying a few rows. For instance, take the `ols.ipynb` file. The first code cell reads:

```
etl = LinearRegressionETL(
    orderbook_path="optiver-realized-volatility-prediction/book_train.parquet/*",
    train_path='optiver-realized-volatility-prediction/train.csv'
)
```

You can change the class parameter to set your own path. The constructor of the ETL class works in the same way for every method. The `AMZN.csv` is needed in the time ordering techniques for some graphical porpouses. It's not essential nor required to run the rest of the code.

Some cell block will also return some output content, like cross-validation results or R-like summaries of linear models. These will be put inside the `summary_*` folders. Please do not delete those, otherwise the code may throw an error. Also, some algorithms will save the results of their analysis, like the final score, in a `.csv` file. These are collected inside the `output` directory.
