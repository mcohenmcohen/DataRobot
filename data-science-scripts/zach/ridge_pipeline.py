import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sagemaker_sklearn_extension.feature_extraction.text import MultiColumnTfidfVectorizer

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Load data and define target
train_file = '~/Downloads/AB_NYC_2019.csv'
data = pd.read_csv(train_file, parse_dates = ['last_review'])
data_train, data_test = train_test_split(data)
del data
TARGET = 'price'

# Split X vs y and train vs test
y_train = data_train[TARGET]
X_train = data_train.drop(TARGET, axis=1)

y_test = data_test[TARGET]
X_test = data_test.drop(TARGET, axis=1)

del data_train, data_test

# Select numeric vs categorical vs text
numeric_selector = make_column_selector(dtype_include=np.number)
categorical_selector = make_column_selector(dtype_include=np.object)

def is_text(x):
    if pd.api.types.is_string_dtype(x):
        pct_rows_with_whitespace = (x.str.count(r'\s') > 0).sum() / x.shape[0]
        return (pct_rows_with_whitespace > .75)
    return False

def text_selector(X):
    return list(X.apply(is_text))

# Make sub-pipelines
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
    ('tfidf', MultiColumnTfidfVectorizer(ngram_range=(1, 2))),
])

# Make pre-processing pipeline
preprocessing_pipeline = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, numeric_selector),
        ('cat', categorical_pipeline, categorical_selector),
        ('txt', text_pipeline, text_selector)
    ]
)

# Make Modeling pipelin
ridge_model = make_pipeline(
    preprocessing_pipeline,
    Ridge(alpha=0.0003, fit_intercept=False)
)

# Fit the model and evaliate on the test set
ridge_model.fit(X_train, y_train)
pred = ridge_model.predict(X_test)
print("Ridge model RMSE: %.3f" % np.sqrt(mean_squared_error(y_test, pred))) #227
