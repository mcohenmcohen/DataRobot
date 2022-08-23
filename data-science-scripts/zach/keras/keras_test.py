###############################################################
#
#   Tests for Keras Estimators
#
#   Author: Zach Deane-Mayer
#
#   Copyright DataRobot Inc, 2018 onwards
#
###############################################################
from __future__ import division, print_function
from io import BytesIO

import numpy as np
import requests
from scipy.sparse import load_npz
from sklearn.metrics import mean_squared_error
import keras
from keras_estimators import (KerasClassifier, KerasRegressor)


def request_raw_bytes(x):
    return BytesIO(requests.get(x, stream=True).raw.read())

X_url = 'https://s3.amazonaws.com/datarobot_public_datasets/fit_X.npz'
y_url = 'https://s3.amazonaws.com/datarobot_public_datasets/fit_y_logged.npy'
X = load_npz(request_raw_bytes(X_url))
y = np.load(request_raw_bytes(y_url))
y = np.exp(y)

rng = np.random.RandomState(42)
indices = rng.permutation(X.shape[0])
split = int(X.shape[0] * .80)
train_idx, test_idx = indices[:split], indices[split:]

deep_model = KerasRegressor(
    epochs=5, loss='mean_squared_logarithmic_error', output_activation='exponential',
    hidden_units=[512, 64, 64],
    hidden_activation='prelu',
    learning_rate=0.003, batch_size=32768, max_batch_size=131072, double_batch_size=True)
deep_model.fit(X[train_idx, :], y[train_idx], verbose=True)
pred = deep_model.predict(X[test_idx, :])
err = np.sqrt(mean_squared_error(np.log1p(pred), np.log1p(y[test_idx])))
print(err)
assert err < 0.40  # 0.39675814
