from typing import Iterable
from analysis import is_outlier
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib
import polars as pl

def plot_hist(data: pl.DataFrame, col:str, dir:str='.', sub: str = ''):
    if sub != '':
        sub = f'_{sub.strip(' _')}'
    ax = sns.histplot(data)
    ax.set_title(col)
    hist_path = f'{dir}/hist_{col}{sub}.jpg'
    pathlib.Path(hist_path).parent.mkdir(parents=True,exist_ok=True)
    plt.savefig(hist_path)
    plt.close()


def plot_feature_histograms(data: pl.DataFrame, dir:str='.', columns: Iterable[str]=None, plot_no_outliers: bool = False):
    if columns is None:
        columns = data.columns
    for col in columns:
        plt.close('all')
        data_col = data.select(pl.col(col))
        if not data_col.dtypes[0].is_numeric():
            continue
        data_col = data_col.drop_nulls()
        if data_col.is_empty():
            continue
        plot_hist(data_col, col, dir)
        if not plot_no_outliers:
            continue
        not_outlier_mask = ~is_outlier(data_col.to_numpy())
        if not_outlier_mask.sum() > 1 and not_outlier_mask.sum() != len(data_col):
            plot_hist(data_col.filter(not_outlier_mask), sub='no_outliers')



def plot_pairplot(data, name: str, dir:str='.', sub: str = ''):
    if sub != '':
        sub = f'_{sub.strip(' _')}'
    
    pair_grid = sns.pairplot(data.to_pandas())
    ax = plt.gca()
    title_str = f'pairplot_{name}{sub}'
    ax.set_title(title_str)
    hist_path = f'{dir}/{title_str}.jpg'
    pathlib.Path(hist_path).parent.mkdir(parents=True,exist_ok=True)
    plt.savefig(hist_path)
    plt.close()
