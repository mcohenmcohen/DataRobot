####################################################################################################
# Setup
####################################################################################################
import numpy as np
import pandas as pd

from category_encoders.ordinal import OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import make_pipeline

from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from vecstack import StackingTransformer

# Load data and define target
train_file = '~/Downloads/AB_NYC_2019.csv'
data = pd.read_csv(train_file)
data_train, data_test = train_test_split(data)
del data
TARGET = 'price'

# Split X vs y and train vs test
y_train = data_train[TARGET]
X_train = data_train.drop(TARGET, axis=1)

y_test = data_test[TARGET]
X_test = data_test.drop(TARGET, axis=1)

del data_train, data_test

####################################################################################################
# Pipeline
####################################################################################################

# Define column types
categorical_columns = list(X_train.select_dtypes(include=np.object).columns.values)
text_columns = 'diag_1_desc'

# Remove text cols from cat cols
categorical_columns = list(set(categorical_columns) - set(text_columns))

# Text ngram modeling
def make_text_ngram_pipeline(column_name, tfidf_kwargs):
    return (make_pipeline(
        FunctionTransformer(
            lambda x: x.fillna('missing').astype(str), validate=False
        ),
        TfidfVectorizer(**tfidf_kwargs),
        StackingTransformer(
            [('r', RidgeClassifier(random_state=0))],
            regression=False, variant='B', n_folds=5,
            stratified=True, shuffle=True)
        ), column_name)

# Preprocessing
tfidf_arg_dict = {
    'ngram_range': (1, 2),
    'norm': 'l2',
    'use_idf': False,
    'min_df': 2,
    'max_df': .5,
    'binary': True,
    'max_features': 200000,
}
preprocessing_pipeline = make_column_transformer(
    make_text_ngram_pipeline('diag_1_desc', tfidf_arg_dict),
    make_text_ngram_pipeline('diag_2_desc', tfidf_arg_dict),
    make_text_ngram_pipeline('diag_3_desc', tfidf_arg_dict),
    (OrdinalEncoder(), categorical_columns),
    remainder='passthrough')
xgb_model = make_pipeline(
    preprocessing_pipeline,
    XGBClassifier(n_estimators=200, max_depth=5, colsample_bytree=.3, learning_rate=.05))
xgb_model.fit(X_train, y_train)

# Evaluate
# DataRobot XGB ~ 0.609
# This XGB ~ .604
pred = xgb_model.predict_proba(X_test)
print("XGB model logloss: %.3f" % log_loss(y_test, pred))
