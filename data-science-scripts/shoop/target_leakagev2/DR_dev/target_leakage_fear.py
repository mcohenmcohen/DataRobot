# EXPERIMENTAL CODE
#
# NOTE1: Mostly borrowed code from FEAR/fear/metapipeline/feature_reduction.py
# NOTE2: EDA2 BECOMES VERY SLOW
# Because of running .fit for each column, oops
# Should consider using this "shap impact importance" approach with leaky customer datasets
#
# Usage:
# - copy contents of this file into ModelingMachine/engine/data_quality/target_leakage.py
# - copy the code block below to eda_multi.py
# 
# # target leakage detection with fear, xgboost, shap
# column_names_training = [
#     col for col in indexed_features.keys() if col != target_name
# ]
# column_var_types = universe_metadata.var_types[:-2]  # ignoring sample_weight and partition_col
# # either training_data or new_df = pq.subset_data(dsn, part_info, 'training')
# # fear_df = pq.subset_data(dsn, part_info, 'training')
# # from common.utilities import forkable_pdb; forkable_pdb.set_trace()
# leakage_type_fear = get_leakage_type_fear(
#     training_data, column_names_training, column_var_types, target_name
# )

import numpy as np
import pandas as pd
import shap

from common.enum import TargetLeakageType, VarTypeCodes
from config.engine import AceLeakageThreshold
from fear.metapipeline import feature_reduction


def get_leakage_type_fear(fear_df, column_names_training, column_var_types, target_name):
    """
    target leakage using fear.
    """
    features_preprocessed = preprocess_features(fear_df, column_var_types)
    y_idx = list(fear_df.columns).index(target_name)
    y = features_preprocessed[:, y_idx]
    X = np.delete(features_preprocessed, y_idx, axis=1)

    model = feature_reduction.XGBEarlyStoppingEstimator()
    model.fit(X, y)

    # shap importance
    model = model.bst
    force_typestr = None
    explainer = shap.TreeExplainer(model, force_typestr=force_typestr)
    shap_values = explainer.shap_values(X)
    shap_impacts = np.mean(np.abs(shap_values), axis=0)
    impacts_with_cols = list(zip(column_names_training, shap_impacts))
    return impacts_with_cols


def preprocess_features(df, column_var_types):  # MODIFIED FROM FEAR
    """ Preprocess FEAR generated features for model training

    Parameters
    ----------
    df : DataFrame
        Data to be preprocessed

    Returns
    -------
    tuple of (DataFrame, ProcessedFeaturesMetadata)
        Processed data ready to be used for training
    """
    #var_types = dict(list(zip(self._metadata.output_columns, self._metadata.output_var_types)))
    cols_list = list(df.columns)
    var_types = dict(list(zip(cols_list, column_var_types)))
    result = []
    # features_metadata = ProcessedFeaturesMetadata()
    for col in df:
        vals = df[col]
        try:
            vals_numeric = pd.to_numeric(vals)
        except ValueError:
            vals_numeric = None
        if var_types.get(col) == 'C' or vals_numeric is None:
            # Use pandas encoder as LabelEncoder is not able to work with mixed dtypes
            uniques = vals.unique()
            encoder = pd.Series(np.arange(len(uniques), dtype=np.int), index=uniques)
            processed = vals.map(encoder)
        else:
            processed = vals_numeric.copy()
            # if self._is_unsupervised:
            #     # IsolationForest can't contain missing or infinite values
            #     processed[pd.isna(processed) | np.isinf(processed.values)] = 0
        result.append(processed)
        # features_metadata.add_processed_feature(col, processed)
    result = np.column_stack(result)
    # assert features_metadata.output_shape == result.shape[1]
    return result
