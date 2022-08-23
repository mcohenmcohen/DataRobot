import numpy as np
import scipy as sp
import tensorflow as tf
import keras as ks
from tensorflow import keras as ks_tf
from scipy.sparse import csr_matrix
print('tf version ' + str(tf.__version__))
print('tf keras version ' + str(tf.keras.__version__))
print('keras keras version ' + str(ks.__version__))
print('numpy version ' + str(np.__version__))
print('scipy version ' + str(sp.__version__))

np.random.seed(42)
X_train = csr_matrix(np.array(
    [[ 36.,   0.,  33.,   0.],
       [  0.,   0.,   0.,   0.],
       [  0.,   0.,  36.,   0.]]
))
y_train = np.array([1, 0, 0])

# Keras, works
model_in = ks.layers.Input(shape=[X_train.shape[1]], sparse=True, dtype='float32')
print(model_in.get_shape())
out = ks.layers.Dense(192, activation='relu')(model_in)
out = ks.layers.Dense(64, activation='relu')(out)
out = ks.layers.Dense(64, activation='relu')(out)
out = ks.layers.Dense(1)(out)
model = ks.models.Model(model_in, out)
model.compile(loss='mean_squared_error', optimizer=ks.optimizers.Adam(lr=3e-3))
model.fit(X_train, y_train)

# TF, doesn't work
from tensorflow import keras as ks_tf
model_in = ks_tf.layers.Input(shape=[X_train.shape[1]], sparse=True, dtype='float32')
out = ks_tf.layers.Dense(192, activation='relu')(model_in)
