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
features_to_rename = {'Pclass': 'Class', 'SibSp': 'Nr_siblings', 'Parch': 'Nr_parents', 'Embarked': 'Port_Embarked'}
df = df.rename(features_to_rename)
#%%
# Identify features
id_col = 'PassengerId'
cabin_col = 'Cabin'
idx_features = [id_col, 'Name', 'Ticket', cabin_col]
social_features = [ 'Class', 'Sex', 'Age', 'Nr_siblings', 'Nr_parents']
travel_features = ['Fare', 'Port_Embarked']
result_features = ['Survived']

features_categorical = ['Class', 'Port_Embarked', 'Survived']
features_numeric = [col for col,dtype in zip(df.columns, df.dtypes) if dtype in pl.NUMERIC_DTYPES and col not in idx_features]
features_numeric_no_categorical = [col for col in features_numeric if col not in features_categorical]
#%%
# Feature engineering
import re
features_from_cabin = ['Deck', 'Room']
travel_features += features_from_cabin

def split_letters_and_numbers(s: str):
    s_last = s.split(' ')[-1]
    return re.findall(r'[a-zA-Z]+', s_last)[0], re.findall(r'[0-9]+', s_last)[0]


def split_cabin_into_letters_and_numbers_struct(cabin: str) -> dict:
    if cabin is None:
        return dict(zip(features_from_cabin, [None]*len(features_from_cabin)))
    return dict(zip(features_from_cabin, split_letters_and_numbers(cabin)))

# Example https://stackoverflow.com/questions/73699500/python-polars-split-string-column-into-many-columns-by-delimiter
df = df.with_columns(
    pl.col(cabin_col).map_elements(
        lambda x: split_cabin_into_letters_and_numbers_struct(x))
       .alias("split_cabin")
        ).unnest("split_cabin")

#%%
# Cast to correct class
features_to_cast = {pl.Categorical: features_categorical} #['Sex', 'Embarked'], pl.Boolean: ['Parch', 'Survived']}
features_to_cast = {}
for cast_type, cast_features in features_to_cast.items():
    for col in cast_features:
        df = df.with_columns(df.select(pl.col(col).cast(cast_type)).to_series().alias(col))
#%%
df
#%%
df.describe()
#%% Feature selection
feature_empty_min_nulls = .5
feature_missing_min_nulls = .2
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
features_categorical = ['Pclass',  'SibSp', 'Parch', 'Embarked', 'Survived']
plot_feature_histograms(df, results_dir+'/hists', hue_variables=features_categorical)

#%%
# Correlations
from plot import plot_pairplot

df_no_nulls = df.drop_nulls()
plot_pairplot(df_no_nulls, name='general', plot_dir=results_dir+'/corrs', sub='strict')

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
