import numpy as np
import pandas as pd

from copy import deepcopy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from scipy.special import expit


# Helper functions
def is_text(x):
    pct_rows_with_whitespace = (x.str.count(r'\s') > 0).sum() / x.shape[0]
    pct_unique = x.nunique() / x.shape[0]
    pct_long = sum(x.str.len() > 32) / x.shape[0]

    heuristic_1 = pct_rows_with_whitespace > .75
    heuristic_2 = pct_unique > .50
    heuristic_3 = pct_long > .25

    return (heuristic_1 and heuristic_2) or (heuristic_1 and heuristic_3)

# Load data and define target
url_10k = 'https://s3.amazonaws.com/datarobot_public_datasets/sas/10k_diabetes.csv'
url_loans = 'https://s3.amazonaws.com/datarobot_public_datasets/10K_Lending_Club_Loans.csv'

data = pd.read_csv(url_10k, encoding = "ISO-8859-1")
TARGET = 'readmitted'

# Split X vs y and train vs test
y = data[TARGET]
X = data.drop(TARGET, axis=1)
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2)
del X, y

# Assume numeric types are numbers
numeric_columns = list(X_train.select_dtypes(include=np.number).columns.values)

# Assume object arrays are categoricals
categorical_columns = list(X_train.select_dtypes(include=np.object).columns.values)

# Assume categoricals with >50% of rows having a space are text too
# Note these columns with BOTH be encoded as a category and as text in the blueprint
text_columns = [col for col in categorical_columns if is_text(X_train[col])]

# Blueprint components
numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

text_pipeline = Pipeline(steps=[
    # SimpleImputer doesn't play nice with tdidf...
    ('imputer', FunctionTransformer(lambda x: x.fillna('missing'), validate=False)),
    ('tfidf', TfidfVectorizer(max_df=2))
    ])

preprocessing = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, numeric_columns),
        ('cat', categorical_pipeline, categorical_columns),
        ] + [('txt ' + x, text_pipeline, x) for x in text_columns]
    )

# Blueprint 1: linear model
linear_model = Pipeline(steps=[
    ('preprocessing', preprocessing),
    ('model', RidgeClassifier(solver='sag')),
])
linear_model.fit(X_train, y_train)
pred_linear = expit(linear_model.decision_function(X_test))
print("ridge model logloss: %.3f" % log_loss(y_test, pred_linear))

# Blueprint 2: GBM model
gbm_model = Pipeline(steps=[
    ('preprocessing', preprocessing),
     # GBMs don't like sparse data, so make it dense with SVD
    ('SVD', TruncatedSVD(n_components=5)),
    ('model', GradientBoostingClassifier()),
])
gbm_model.fit(X_train, y_train)
pred_gbm = gbm_model.predict_proba(X_test)
print("GBM model logloss: %.3f" % log_loss(y_test, pred_gbm))
