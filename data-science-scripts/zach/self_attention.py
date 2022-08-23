"""keras interaction models"""
import os
import requests
import keras.backend as K

import numpy as np
from io import BytesIO
from scipy.sparse import load_npz, csr_matrix
from sklearn.metrics import mean_squared_error
from keras.layers import Input, Dense
from keras.layers import TimeDistributed, RepeatVector
from keras.layers import Dot, Add, Reshape, Flatten, Activation
from keras.layers import Conv1D, BatchNormalization, Multiply
from keras.activations import softmax
from keras.models import Model
from keras.optimizers import Adam

os.environ['OMP_NUM_THREADS'] = '6'

# Load Data
def request_raw_bytes(x):
    return BytesIO(requests.get(x, stream=True).raw.read())

X_url = 'https://s3.amazonaws.com/datarobot_public_datasets/fit_X.npz'
y_url = 'https://s3.amazonaws.com/datarobot_public_datasets/fit_y_logged.npy'
X = load_npz(request_raw_bytes(X_url))
y = np.load(request_raw_bytes(y_url))

rng = np.random.RandomState(42)
indices = rng.permutation(X.shape[0])
split = int(X.shape[0] * 0.80)
train_idx, test_idx = indices[:split], indices[split:]

# Define Model
N = X.shape[1]
dim = 128
model_in = Input((N,), name='input', sparse=True)
out = model_in

queries = Dense(dim, name='queries')(out)
keys = Dense(dim, name='keys')(out)
queries = Reshape((-1, 1), name='queries-reshaped')(queries)
keys = Reshape((-1, 1), name='keys-reshaped')(keys)

out = Dot(axes=2, name='dot')([queries, keys])

attention = Dense(dim, activation='softmax', name='attention')(out)
values = Dense(dim, name='values', activation='linear')(out)
out = Multiply(name='mult')([attention, values])
out = Flatten(name='flat')(out)

out = Dense(1, name='output')(out)
pass_through = Dense(1, name='passthrough')(model_in)

out = Add(name='add')([pass_through, out])
out = Activation('linear', name='link')(out)

model = Model(model_in, out)
model.compile(Adam(0.003), loss='mean_squared_error')
model.summary()

# Fit model
model.fit(X[train_idx, :], y[train_idx], verbose=True, batch_size=4096)

# Evaluate model - benchmark 0.39675814
pred = model.predict(X[test_idx, :])
err = np.sqrt(mean_squared_error(pred, y[test_idx]))
print(err)
assert err < 0.40
