#%% [Markdown]
# # Titanic
# We will study the titanic Kaggle dataset found in: https://www.kaggle.com/competitions/titanic
# This will have 3 parts: the EDA (Exploratory Data Analysis), CDA (Confirmatory Data Analysis) & Machine Learning / Deep Learning Model training
#%%
data_dir = 'data'
results_dir = 'results'
test_data_file = 'test.csv'
train_data_file = 'train.csv'
test_data_filepath = f'{data_dir}/{test_data_file}'
train_data_filepath = f'{data_dir}/{train_data_file}'
#%% [Markdown]
# ## EDA
# Basic stats, histograms, correlations & grouping to describe the data and the inherent relationships
#%%
# Stats
import polars as pl
df = pl.read_csv(test_data_filepath)
df
#%%
df.columns
#%%
df.dtypes
#%%
df.describe()
#%%
import numpy as np
def is_outlier(points, thresh=3.5):
    """
    From: https://stackoverflow.com/questions/11882393/matplotlib-disregard-outliers-when-plotting

    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

#%%
def plot_hist(data, sub: str = ''):
    if sub != '':
        sub = f'_{sub.strip(' _')}'
    ax = sns.histplot(data_col)
    ax.set_title(col)
    hist_path = f'{results_dir}/hist_{col}{sub}.jpg'
    pathlib.Path(hist_path).parent.mkdir(parents=True,exist_ok=True)
    plt.savefig(hist_path)
    plt.close()
#%%
# Histograms
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib

plot_no_outliers = False
for col in df.columns:
    plt.close('all')
    data_col = df.select(pl.col(col))
    if not data_col.dtypes[0].is_numeric():
        continue
    data_col = data_col.drop_nulls()
    if data_col.is_empty():
        continue
    plot_hist(data_col)
    if not plot_no_outliers:
        continue
    not_outlier_mask = ~is_outlier(data_col.to_numpy())
    if not_outlier_mask.sum() > 1 and not_outlier_mask.sum() != len(data_col):
        plot_hist(data_col.filter(not_outlier_mask), sub='no_outliers')

#%%
#%%
#%%
#%%
