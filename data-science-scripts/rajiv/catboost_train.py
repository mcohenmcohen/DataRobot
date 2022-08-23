import pandas as pd
import numpy as np
import os
import shutil
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score

from catboost_demo.src.utils import normalize_text
from catboost_demo.src.common_strings import *

def induce_drift(input_path,
                 output_path,
                 random_seed):
    df_drifted = pd.read_csv(input_path)
    readmitted = df_drifted[df_drifted[TARGET] == False]
    lower_socioeco = readmitted.sample(frac=0.8, replace=False, random_state=random_seed)
    lower_socioeco[TARGET] = True
    number_diagnoses_drift = np.random.normal(loc=df_drifted[NUM_DIAG].median() + 20.,
                                              scale=5,
                                              size=lower_socioeco[NUM_DIAG].shape)

    lower_socioeco[NUM_DIAG] = number_diagnoses_drift.round().astype(int)
    number_inpatient_drift = np.random.normal(loc=df_drifted[NUM_INPATIENT].median() + 20.,
                                              scale=2.,
                                              size=lower_socioeco[NUM_INPATIENT].shape)
    lower_socioeco[NUM_INPATIENT] = number_inpatient_drift.round().astype(int)
    cardiovascular_keywords = ['Coronoary', 'Congestive', 'heart', 'infarcation']
    diag1 = df_drifted[[DIAG1, DIAG1_DESC]]
    severe_diagnoses = diag1[DIAG1_DESC].str.contains('|'.join(cardiovascular_keywords))
    diag1['severe'] = severe_diagnoses
    bad_diseases = diag1[diag1['severe'] == True][[DIAG1, DIAG1_DESC]].drop_duplicates()
    diag1_drift = bad_diseases.sample(n=lower_socioeco.shape[0], replace=True, random_state=random_seed)
    lower_socioeco[DIAG1] = diag1_drift[DIAG1].values
    lower_socioeco[DIAG1_DESC] = diag1_drift[DIAG1_DESC].values
    df_drifted.update(lower_socioeco)
    df_drifted.to_csv(output_path, index=False)


def diabetes_train_test_split(df,
                              train_size=0.8,
                              test_size=0.2):
    readmitting = df[TARGET] == POSITIVE_CLASS_LABEL
    df_train, df_test = train_test_split(df, train_size=train_size, random_state=random_seed, stratify=readmitting)
    df_train = df_train.drop(REMOVE_COLUMNS, axis=1)
    y_train = df_train.pop(TARGET)
    X_train = df_train
    df_test = df_test.drop(REMOVE_COLUMNS, axis=1)
    y_test = df_test.pop(TARGET)
    X_test = df_test

    # Validation Set
    readmitting_train = y_train == POSITIVE_CLASS_LABEL
    X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                      y_train,
                                                      test_size=test_size,
                                                      random_state=random_seed,
                                                      stratify=readmitting_train)
    return X_train, y_train, X_val, y_val, X_test, y_test

def train_catboost(X_train,
                   y_train,
                   X_val,
                   y_val):
    numeric_features = list(X_train.select_dtypes(include=np.number).columns.values)
    categorical_features = list(set(X_train.columns) - set(numeric_features))
    cat_feature_indices = list(range(0, len(categorical_features)))

    numeric_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
            ('scaler', StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing'))
        ]
    )

    text_transformer = Pipeline(
        steps=[
            ('normalize', FunctionTransformer(normalize_text, validate=False)),
            ('tfidf', TfidfVectorizer()),
            ('svd', TruncatedSVD())
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
                         ('cat', categorical_transformer, categorical_features),
                         ('num', numeric_transformer, numeric_features)
                     ]
                     # + [('txt ' + x, text_transformer, x) for x in TEXT_FEATURES]
    )

    preprocessor.fit(X_train)
    X_val = preprocessor.transform(X_val)

    catboost_classifier = Pipeline(
        steps=[('preprocessor', preprocessor),
               ('classifier', CatBoostClassifier(iterations=250,
                                                 loss_function='Logloss',
                                                 verbose=True))]
    )

    val = np.hstack([X_val, y_val.values.reshape(-1, 1)])
    val_pool = Pool(data=val[:, :-1], label=val[:, -1].astype('float'), cat_features=cat_feature_indices)

    y_train = y_train.astype(np.float)
    catboost_classifier.fit(X_train,
                            y_train,
                            classifier__cat_features=cat_feature_indices,
                            classifier__eval_set=val_pool)

    return catboost_classifier


if __name__ == '__main__':
    random_seed = 0
    data_store_path = '../../data_store'
    df = pd.read_csv(data_store_path + '/10k_diabetes.csv')
    df['patient_id'] = df.index
    X_train, y_train, X_val, y_val, X_test, y_test = diabetes_train_test_split(df)
    Xy_train = pd.concat([X_train, y_train], axis=1)
    Xy_train['Fold'] = 'Training'
    Xy_val = pd.concat([X_val, y_val], axis=1)
    Xy_val['Fold'] = 'Validation'
    Xy_test = pd.concat([X_test, y_test], axis=1)
    Xy_test['Fold'] = 'Test'
    Xy_test = Xy_test.drop('readmitted', axis=1)
    Xy_test.to_csv(os.path.join(data_store_path, '10k_diabetes_test.csv'), index=False)
    train_val_df = pd.concat([Xy_train, Xy_val], axis=0)
    train_val_df.to_csv(data_store_path + '/10k_diabetes_train_val.csv', index=False)

    catboost_classifier = train_catboost(X_train,
                                         y_train,
                                         X_val,
                                         y_val)

    joblib.dump(catboost_classifier, '../catboost_trained.pkl')

    test_scores = catboost_classifier.predict_proba(X_test)
    test_auc = roc_auc_score(y_test, test_scores[:, 1])
    print(test_auc)

    shutil.rmtree('catboost_info')
    #
    # induce_drift(input_path=data_store_path + '/10k_diabetes.csv',
    #              output_path=data_store_path + '/10k_diabetes_drifted.csv',
    #              random_seed=random_seed)
    #
    # X_train, y_train, X_val, y_val, X_test, y_test = diabetes_train_test_split(input_path=data_store_path + '/10k_diabetes_drifted.csv',
    #                                                                            output_train_path=data_store_path + '/10k_diabetes_drifted_train.csv',
    #                                                                            output_test_path=data_store_path + '/10k_diabetes_drifted_test.csv')
    #
    # Xy_train = pd.concat([X_train, y_train], axis=1)
    # Xy_train['Fold'] = 'Training'
    # Xy_val = pd.concat([X_val, y_val], axis=1)
    # Xy_val['Fold'] = 'Validation'
    # train_val_df = pd.concat([Xy_train, Xy_val], axis=0)
    # train_val_df.to_csv(data_store_path + '/10k_diabetes_drifted_train_val.csv')
    #
    # train_catboost(X_train,
    #                y_train,
    #                X_val,
    #                y_val,
    #                X_test,
    #                y_test,
    #                artifact_path='../drifted_catboost_trained.pkl')


