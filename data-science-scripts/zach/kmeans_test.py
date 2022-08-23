import numpy as np
import pandas as pd
import time

from ModelingMachine.engine.modeling_data import create_modeling_data
from ModelingMachine.engine.modeling_data import create_task_data

from ModelingMachine.engine.tasks2.na_imputation import Numeric_impute_V2 as PNI2
from ModelingMachine.engine.tasks2.standardize import Standardize as ST
from ModelingMachine.engine.tasks2.reduction_transformers import mmTruncatedSVD as SVD
from ModelingMachine.engine.tasks2.cluster.k_means import KMeansMulticModeler as KMEANS_CL
from ModelingMachine.engine.tasks2.cluster.hdbscan import HDBSCANModeler as HDBSCAN_CL

df = pd.read_csv('/home/datarobot/bigger_cluster_data.csv')

modeling_data = create_modeling_data(
    X=df.values,
    Y=np.ones(df.shape[0]),
    target=None,
    weights=None,
    sample_weight=None,
    balanced_weights=None,
    original_sample_weight=None,
    row_index=df.index.values,
    user_partition=None,
    mask=None,
    colnames=list(df.columns),
    coltypes=[],
    metadata={},
)

task_data = create_task_data(
    cv_method='RandomCV',
    fill_cv=True,
    inputs=None,
    prime_alpha=False,
    refresh=None,
    method=None,
    first_partition=None,
    compute_hotspots=True,
)

# kmeans
t0 = time.time()
data_preprocessed = modeling_data
data_preprocessed = PNI2('dtype=float32').fit_transform(data_preprocessed, task_data)
data_preprocessed = ST('dtype=float32;fm=True').fit_transform(data_preprocessed, task_data)
data_preprocessed = SVD('dtype=float32').fit_transform(data_preprocessed, task_data)
predictions = KMEANS_CL('k=3;pp_stk=0;t_clustering_score_on_training=True').fit_predict(data_preprocessed, task_data)
t1 = time.time()
assert predictions.X.shape[0] == 3000000
assert predictions.X.shape[1] == 3
print(t1 - t0)

# HDBSCAN
t0 = time.time()
predictions = HDBSCAN_CL('cluster_selection_epsilon=0.2;t_clustering_score_on_training=True').fit_predict(data_preprocessed, task_data)
t1 = time.time()
assert predictions.X.shape[0] == 3000000
assert predictions.X.shape[1] == 3
print(t1 - t0)
