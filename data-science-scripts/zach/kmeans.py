####################################################################################################
# Setup
####################################################################################################
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sagemaker_sklearn_extension.feature_extraction.text import MultiColumnTfidfVectorizer

from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

url = 'https://s3.amazonaws.com/datarobot_public_datasets/sas/10k_diabetes.csv'
X_train = pd.read_csv(url)

####################################################################################################
# Cluster
####################################################################################################

numeric_selector = make_column_selector(dtype_include=np.number)
categorical_selector = make_column_selector(dtype_include=np.object)

def is_text(x):
    if pd.api.types.is_string_dtype(x):
        pct_rows_with_whitespace = (x.str.count(r'\s') > 0).sum() / x.shape[0]
        return (pct_rows_with_whitespace > .75)
    return False

def text_selector(X):
    return list(X.apply(is_text))

numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

text_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('tfidf', MultiColumnTfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=.8)),
])

preprocessing_pipeline = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, numeric_selector),
        ('cat', categorical_pipeline, categorical_selector),
        ('txt', text_pipeline, text_selector)
    ]
)

# Model
kmean_model = make_pipeline(
    preprocessing_pipeline,
    TruncatedSVD(n_components=5),
    KMeans()
)
clusters = kmean_model.fit_predict(X_train)
print(np.unique(clusters, return_counts=True))
print(kmean_model.steps[-1][1] .cluster_centers_)
