from typing import Iterable
from analysis import is_outlier
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib
import polars as pl



def plot_feature_histograms(data: pl.DataFrame, plot_dir:str='.', columns: Iterable[str]=None, hue_variables: Iterable[str]=None, plot_no_outliers: bool = False):
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

        # Plot normal histogram
        plot_hist(data_col, col, plot_dir=plot_dir)

        # Plot histogram no outliers
        if plot_no_outliers:
            not_outlier_mask = ~is_outlier(data_col.to_numpy())
            if not_outlier_mask.sum() > 1 and not_outlier_mask.sum() != len(data_col):
                data_col = data_col.filter(not_outlier_mask)
                plot_hist(data_col, col=col, plot_dir=plot_dir, sub='no_outliers')
        
        # Plot histogram by categoricals
        if hue_variables is not None:
            for hue_col in hue_variables:
                if col != hue_col:
                    plot_hist(data, col=col, hue_col=hue_col, plot_dir=plot_dir)


def plot_hist(data: pl.DataFrame, col:str, hue_col:str=None, plot_dir:str='.', sub: str = ''):
    if sub != '':
        sub = f'_{sub.strip(' _')}'
    if hue_col is not None:
        hue_str = hue_col
        if not isinstance(hue_col, str) and isinstance(hue_col, Iterable):
            hue_str = '_'.join(c for c in hue_col)
        sub = f'{sub}_by_{hue_str}'
    ax = sns.histplot(data, x=col, hue=hue_col, multiple='stack')
    ax.set_title(col)
    fpath = f'{plot_dir}/hist_{col}{sub}.jpg'
    pathlib.Path(fpath).parent.mkdir(parents=True,exist_ok=True)
    plt.savefig(fpath)
    plt.close()


def plot_pairplot(data, name: str, plot_dir:str='.', sub: str = ''):
    if sub != '':
        sub = f'_{sub.strip(' _')}'
    
    pair_grid = sns.pairplot(data.to_pandas())
    ax = plt.gca()
    title_str = f'pairplot_{name}{sub}'
    ax.set_title(title_str)
    hist_path = f'{plot_dir}/{title_str}.jpg'
    pathlib.Path(hist_path).parent.mkdir(parents=True,exist_ok=True)
    plt.savefig(hist_path)
    plt.close()
