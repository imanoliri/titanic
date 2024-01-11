from typing import Iterable, List, Tuple, Union
from analysis import is_outlier
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib
import polars as pl
from sklearn.decomposition import PCA
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


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


def plot_feature_2d_histograms(data: pl.DataFrame, columns: Iterable[Iterable[str]], plot_dir:str='.', plot_no_outliers: bool = True):
    for cols in columns:
        plt.close('all')
        data_col = data.select(pl.col(cols))
        if not all(dt.is_numeric() or dt == pl.Categorical for dt in data_col.dtypes):
            continue
        data_col = data_col.drop_nulls()
        if data_col.is_empty():
            continue

        # Plot normal histogram
        plot_2d_hist(data_col, cols=cols, plot_dir=plot_dir)

        # Plot histogram no outliers
        if plot_no_outliers and all(dt.is_numeric() for dt in data_col.dtypes):
            data_vals = data_col.to_numpy()
            not_outlier_mask = np.bitwise_and(~is_outlier(data_vals),~is_outlier(data_vals))
            if not_outlier_mask.sum() > 1 and not_outlier_mask.sum() != len(data_col):
                data_col = data_col.filter(not_outlier_mask)
                plot_2d_hist(data_col, cols=cols, plot_dir=plot_dir, sub='no_outliers')


def plot_2d_hist(data: pl.DataFrame, cols: Iterable[str], plot_dir:str='.', sub: str = ''):
    if sub != '':
        sub = f'_{sub.strip(' _')}'
    cols_str = '_'.join(cols)

    ax = sns.histplot(data, x=cols[0], y=cols[1], multiple='stack')
    ax.set_title(cols_str)
    fpath = f'{plot_dir}/hist_{cols_str}{sub}.jpg'
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
    fpath = f'{plot_dir}/{title_str}.jpg'
    pathlib.Path(fpath).parent.mkdir(parents=True,exist_ok=True)
    plt.savefig(fpath)
    plt.close()


def plot_correlations(corr: pd.DataFrame, name: str, plot_dir:str='.', sub: str = '', corner: bool = True):
    # https://seaborn.pydata.org/examples/many_pairwise_correlations.html

    title_str = f'correlations_{name}_{sub}'
    path = f'{plot_dir}/{title_str}'
    
    corr.to_csv(f'{path}.csv')
    corr_plot = corr
    if corner:
        corr_plot = corr.where(~np.triu(np.ones(corr.shape)).astype(bool))

    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_plot,
                cmap=cmap,
                vmin=-1,
                vmax=1,
                center=0,
                square=True,
                annot=True,
                linewidths=.5,
                cbar_kws={"shrink": .5})
    ax = plt.gca()
    ax.set_title(title_str)
    
    fpath = f'{plot_dir}/{title_str}.jpg'
    pathlib.Path(fpath).parent.mkdir(parents=True,exist_ok=True)
    plt.savefig(fpath)
    plt.close()
    plt.close('all')


def correlations_autoreport(corr: pd.DataFrame, name: str, plot_dir:str='.', sub: str = '',
                    high_corr_threshold: float = 0.5,
                    low_corr_threshold: float = 0.1,
                    corrs_for_each_param: int = 3):    
    autoreport_lines = []
    df_corr_all = corr.stack().to_frame()
    df_corr_all.index.names = ['var1', 'var2']
    df_corr_all.columns = ['correlation']
    df_corr_all['abs_correlation'] = df_corr_all.abs()
    mask_coincident_variables = np.array(
        [v1 in v2 or v2 in v1 for v1, v2 in df_corr_all.index.values])
    df_corr_all = df_corr_all.loc[~mask_coincident_variables]

    # Only the lower triangular
    corr_tri = corr.where(~np.triu(np.ones(corr.shape)).astype(bool))
    df_corr = corr_tri.stack().to_frame()
    df_corr.index.names = ['var1', 'var2']
    df_corr.columns = ['correlation']
    mask_coincident_variables = np.array(
        [v1 in v2 or v2 in v1 for v1, v2 in df_corr.index.values])
    df_corr = df_corr.loc[~mask_coincident_variables]
    df_corr['abs_correlation'] = df_corr.abs()
    df_corr = df_corr.sort_values('abs_correlation', ascending=False)
    df_corr['high_corr'] = df_corr['abs_correlation'] > high_corr_threshold
    df_corr['low_corr'] = df_corr['abs_correlation'] < low_corr_threshold

    autoreport_lines.append('High correlations')
    autoreport_lines += [
        f'{v1} - {v2}: {round(row.correlation,2)}'
        for (v1, v2), row in df_corr.loc[df_corr['high_corr']].iterrows()
    ]

    autoreport_lines.append('\nLow correlations')
    autoreport_lines += [
        f'{v1} - {v2}: {round(row.correlation,2)}'
        for (v1, v2), row in df_corr.loc[df_corr['low_corr']].iterrows()
    ]

    autoreport_lines.append('\nHighest & lowest correlations per variable')
    df_corr_high_low = df_corr_all.reset_index().groupby('var1').apply(
        lambda x: x.sort_values('abs_correlation', ascending=False
                                ).iloc[:, :3].values)
    autoreport_lines += [
        f'{v1} - {v2}: {round(corr,2)}' for row in df_corr_high_low.values
        for v1, v2, corr in np.array(
            [*row[:corrs_for_each_param], *row[-corrs_for_each_param:]])
    ]


    title_str = f'correlations_{name}_{sub}'
    path = f'{plot_dir}/{title_str}'
    fpath = f'{path}_autoreport.txt'
    pathlib.Path(fpath).parent.mkdir(parents=True,exist_ok=True)
    with open(fpath, 'w') as f:
        f.write('\n'.join(autoreport_lines))


def plot_pca(pca: PCA, columns: Iterable[str], name: str, plot_dir: str='.', sub: str = ''):
    if sub != '':
        sub = f'_{sub.strip(' _')}'
    title_str = f'pca_{name}{sub}'
    fpath = f'{plot_dir}/{title_str}'
    pathlib.Path(fpath).parent.mkdir(parents=True,exist_ok=True)

    comps = pca_component_matrix(pca, columns)
    comps.write_csv(f'{fpath}.csv')

    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    comps_to_plot = comps.drop(columns=['explained_variance', 'comp'])
    sns.heatmap(comps_to_plot,
                xticklabels = comps_to_plot.columns,
                cmap=cmap,
                vmin=-1,
                vmax=1,
                center=0,
                square=True,
                annot=True,
                linewidths=.5,
                cbar_kws={"shrink": .5})
    ax.set_title(title_str.replace(' ', '_'))

    plt.savefig(f'{fpath}.jpg', transparent=False)
    plt.close('all')

    autoreport_lines = pca_autoreport(pca, columns)

    with open(f'{fpath}_autoreport.txt', 'w') as f:
        f.write('\n'.join(autoreport_lines))


def pca_component_matrix(pca: PCA, columns: Iterable[str]) -> pl.DataFrame:
    comps = pca.components_
    comps = pl.DataFrame(comps,
                         schema=columns)
    comps = comps.select(pl.col(col).round(2) for col in comps.columns)
    comps = comps.insert_column(0, pl.Series('comp',[f'Comp_{i+1}' for i in range(comps.shape[0])]))
    return comps.insert_column(0, pl.Series('explained_variance',
                 (100 * pca.explained_variance_ratio_).round(0)))


def pca_autoreport(pca: PCA, columns: Iterable[str], top_comps:int = 5, top_factors = 5) -> List[str]:
    autoreport_lines = []
    autoreport_lines.append('Top components are:\n')
    
    comps = pca_component_matrix(pca, columns).to_pandas().iloc[:,2:]
    comps_top = comps.iloc[:top_comps]
    top_comps_and_factors = [
        comp.loc[comp.abs().sort_values(
            ascending=False).iloc[:top_factors].index]
        for r, comp in comps_top.iterrows()
    ]
    lines_top_comps = [
        '; '.join(f'{k}: {v}' for k, v in cf.to_dict().items())
        for cf in top_comps_and_factors
    ]
    autoreport_lines += [
        f'{c} --> {nl}'
        for c, nl in zip(comps.index[:top_comps], lines_top_comps)
    ]
    return autoreport_lines

