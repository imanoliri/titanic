#%% [Markdown]
# # Titanic
# We will study the titanic Kaggle dataset found in: https://www.kaggle.com/competitions/titanic
# This will have 4 parts: basic descriptive analysis, EDA (Exploratory Data Analysis), CDA (Confirmatory Data Analysis) & Machine Learning / Deep Learning / Model training
#%%
data_dir = 'data'
results_dir = 'results'
test_data_file = 'test.csv'
train_data_file = 'train.csv'
survived_data_file = 'gender_submission.csv'
test_data_filepath = f'{data_dir}/{test_data_file}'
train_data_filepath = f'{data_dir}/{train_data_file}'
survived_data_filepath = f'{data_dir}/{survived_data_file}'
#%% [Markdown]
# ## Data description
#%%
# Stats
import polars as pl
df = pl.read_csv(test_data_filepath)
df_survived = pl.read_csv(survived_data_filepath)
df = df.join(df_survived, on='PassengerId')
df
#%%
# Rename features
features_to_rename = {'Pclass': 'Class', 'SibSp': 'Nr_siblings_or_spouses', 'Parch': 'Nr_parents_or_children', 'Embarked': 'Port_Embarked'}
df = df.rename(features_to_rename)
#%%
# Identify features
id_col = 'PassengerId'
cabin_col = 'Cabin'
price_col = 'Fare'
nr_relatives_col = 'Nr_total_relatives'
nr_sibl_spou_col = 'Nr_siblings_or_spouses'
nr_parent_child_col = 'Nr_parents_or_children'
idx_features = [id_col, 'Name', 'Ticket', cabin_col]
social_features = [ 'Class', 'Sex', 'Age', nr_sibl_spou_col, nr_parent_child_col, nr_relatives_col]
travel_features = [price_col, 'Port_Embarked']
result_features = ['Survived']

features_categorical = ['Class', 'Port_Embarked', 'Survived']+['Sex']
def numeric_columns (df: pl.DataFrame) -> list:
    return [col for col,dtype in zip(df.columns, df.dtypes) if dtype in pl.NUMERIC_DTYPES and col not in idx_features]

features_numeric = numeric_columns(df)
features_numeric_no_categorical = [col for col in features_numeric if col not in features_categorical]
#%%
# Feature engineering
#%%
# Add total number of relatives
df = df.with_columns((pl.col(nr_sibl_spou_col) + pl.col(nr_parent_child_col)).alias(nr_relatives_col))
#%%
# Divide cabin str into features
import re
features_from_cabin = ['Deck', 'Room']
nr_cabins_col = 'Nr_cabins'
travel_features += features_from_cabin + [nr_cabins_col]

def split_letters_and_numbers(s: str):
    s_last = s.split(' ')[-1]
    return re.findall(r'[a-zA-Z]+', s_last)[0], re.findall(r'[0-9]+', s_last)[0]


def split_cabin_into_letters_and_numbers_struct(cabin: str) -> dict:
    if cabin is None:
        return dict(zip(features_from_cabin, [None]*len(features_from_cabin)))
    return dict(zip(features_from_cabin, split_letters_and_numbers(cabin)))

# Example https://stackoverflow.com/questions/73699500/python-polars-split-string-column-into-many-columns-by-delimiter

# Divide into letter and number
df = df.with_columns(
    pl.col(cabin_col).map_elements(split_cabin_into_letters_and_numbers_struct)
       .alias("split_cabin")
        ).unnest("split_cabin")

# get nr of cabins
df = df.with_columns(pl.col(cabin_col).map_elements(lambda cabin: len(cabin.split(' '))).alias(nr_cabins_col))
df = df.with_columns(pl.col(nr_cabins_col).fill_null(1))

# correct price per nr of cabins
df = df.with_columns(pl.col(price_col) / pl.col(nr_cabins_col))

#%%
# Cast to correct class
features_to_cast = {pl.Categorical: features_categorical} #, pl.Boolean: ['Parch', 'Survived']}
for cast_type, cast_features in features_to_cast.items():
    for col in cast_features:
        if df.select(pl.col(col)).dtypes[0] != pl.Utf8:
            continue
        df = df.with_columns(df.select(pl.col(col).cast(cast_type)).to_series().alias(col))
#%%
df
#%%
df.describe()
#%% Feature selection
feature_empty_min_nulls = 0.5
feature_missing_min_nulls = 0.2
empty_features = []
features_with_missing = []
for col in df.columns:
    feature_null_ratio = df.select(pl.col(col)).null_count().to_numpy()[0,0] / df.shape[0]
    if feature_null_ratio >= feature_empty_min_nulls:
        empty_features.append(col)
        continue
    if feature_null_ratio >= feature_missing_min_nulls:
        features_with_missing.append(col)
empty_features
#%%
features_with_missing
#%% Remove empty features
df = df.select(pl.col(c for c in df.columns if c not in empty_features))
#%% Impute missing features (with median)
for col in features_with_missing:
   df_col = df.select(pl.col(col))
   df_col = df_col.fill_null(df_col.median())
   df = df.with_columns(df_col.to_series().alias(col))

#%% [Markdown]
# ## EDA
# Basic stats, histograms, correlations & grouping to describe the data and the inherent relationships

#%%
# Histograms
from plot import plot_feature_histograms
plot_feature_histograms(df, results_dir+'/hists', hue_variables=features_categorical)

#%%
# 2D Histograms
from plot import plot_feature_2d_histograms
from itertools import combinations
# social_2d_variables = [('Age', 'Sex'), ('Class', 'Sex'), ('Age', 'Class')]
# travel_2d_variables = [('Fare', 'Port_Embarked')]
# variables_2d_histograms = social_2d_variables + travel_2d_variables
variables_2d_histograms = list(cols for cols in combinations(social_features+travel_features, 2) if not any(c in empty_features for c in cols))
plot_feature_2d_histograms(df, columns=variables_2d_histograms, plot_dir=results_dir+'/hists_2d', plot_no_outliers=False)

#%%
# Correlations
from plot import plot_pairplot, plot_correlations, correlations_autoreport

df_no_idx = df.select(pl.col(c for c in df.columns if c != id_col))
df_no_nulls = df_no_idx.drop_nulls()
plot_pairplot(df_no_nulls, name='general', plot_dir=results_dir+'/corrs', sub='strict')
# TODO: remove non numeric!!
df_numeric_no_nulls = df_no_nulls.select(pl.col(numeric_columns(df_no_nulls)))
df_corrs = df_numeric_no_nulls.to_pandas().corr().round(2)
plot_correlations(df_corrs, name='general', plot_dir=results_dir+'/corrs', sub='strict')
correlations_autoreport(df_corrs, name='general', plot_dir=results_dir+'/corrs', sub='strict')

#%%
# PCA
import sklearn
from plot import plot_pca

pca_n_components = 5
pca = sklearn.decomposition.PCA(n_components=min(pca_n_components, df_no_nulls.shape[1]))
pca.fit(df_no_nulls.select(features_numeric))
pca_no_category = sklearn.decomposition.PCA(n_components=min(pca_n_components,len(features_numeric_no_categorical)))
pca_no_category.fit(df_no_nulls.select(features_numeric_no_categorical))
plot_pca(pca, columns=features_numeric, name='general', plot_dir=results_dir+'/corrs')
plot_pca(pca_no_category, columns=features_numeric_no_categorical, name='general', plot_dir=results_dir+'/corrs', sub='no_categoricals')
#%%
#%%
# LDA
#%%
#%%
