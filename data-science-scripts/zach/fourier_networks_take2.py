from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam, SGD, Nadam
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Dot, Add, Concatenate, Multiply
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras import regularizers
import pandas as pd
from keras.constraints import non_neg, min_max_norm
from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import RandomNormal
from sklearn.preprocessing import MinMaxScaler
from keras.utils import plot_model
from matplotlib.pyplot import imshow
import numpy as np
from scipy.fftpack import fft, ifft
%matplotlib osx

# sin activation
def sin(x):
  return K.sin(x)
get_custom_objects().update({'sin': Activation(sin)})

# Cosine activation
def cos(x):
  return K.cos(x)
get_custom_objects().update({'cos': Activation(cos)})

# Quad activation
def quad(x):
  return x ** 2
get_custom_objects().update({'quad': Activation(quad)})

# train = pd.read_csv('https://s3.amazonaws.com/datarobot_public_datasets/time_series/hyndman_turkish_electricity_demand_train.csv')
# test = pd.read_csv('https://s3.amazonaws.com/datarobot_public_datasets/time_series/hyndman_turkish_electricity_demand_test.csv')

train = pd.read_csv('https://s3.amazonaws.com/datarobot_public_datasets/time_series/weekly_earth_co2_train.csv')
test = pd.read_csv('https://s3.amazonaws.com/datarobot_public_datasets/time_series/weekly_earth_co2_test.csv')

# train = pd.read_csv('https://s3.amazonaws.com/datarobot_public_datasets/time_series/internet_time_series_train.csv')
# test = pd.read_csv('https://s3.amazonaws.com/datarobot_public_datasets/time_series/internet_time_series_test.csv')

# train = pd.read_csv('https://s3.amazonaws.com/datarobot_public_datasets/time_series/wunderground_Chicago_actual_max_temp_train.csv')
# test = pd.read_csv('https://s3.amazonaws.com/datarobot_public_datasets/time_series/wunderground_Chicago_actual_max_temp_test.csv')

# train = pd.read_csv('https://s3.amazonaws.com/datarobot_public_datasets/time_series/iso_ne_hourly_load_train.csv')
# test = pd.read_csv('https://s3.amazonaws.com/datarobot_public_datasets/time_series/iso_ne_hourly_load_test.csv')

# train = pd.read_csv('https://s3.amazonaws.com/datarobot_public_datasets/time_series/facebook_peyton_manning_train.csv') # Bad
# test = pd.read_csv('https://s3.amazonaws.com/datarobot_public_datasets/time_series/facebook_peyton_manning_test.csv')

# train = pd.read_csv('https://s3.amazonaws.com/datarobot_public_datasets/time_series/facebook_retail_train.csv')
# test = pd.read_csv('https://s3.amazonaws.com/datarobot_public_datasets/time_series/facebook_retail_test.csv') # weird offset

# train = pd.read_csv('https://s3.amazonaws.com/datarobot_public_datasets/time_series/fpp_antidiabetic_drugs_train.csv')
# test = pd.read_csv('https://s3.amazonaws.com/datarobot_public_datasets/time_series/fpp_antidiabetic_drugs_test.csv')

# Train set
# X = np.linspace(0.0, 25, 10000).reshape(-1, 1)
# Y = np.sin(1*X) + np.cos(2*X) + np.cos(3*X) + 0.3*X
X = np.arange(train.shape[0])
Y = train[['y']].values
X_dummy = np.zeros(X.shape, dtype=np.int8)

# Test Set
# X2 = np.linspace(25, 50, 10000).reshape(-1, 1)
# Y2 = np.sin(1*X2) + np.cos(2*X2) + np.cos(3*X2) + 0.3*X2
X2 = X.max() + np.arange(test.shape[0])
Y2 = test[['y']].values
X2_dummy = np.zeros(X2.shape, dtype=np.int8)

# Scale
SCALE_MAX = 1000
scale_x = MinMaxScaler((0,SCALE_MAX))
scale_x.fit(X)

scale_y = MinMaxScaler((0,100))
scale_y.fit(Y)

X = scale_x.transform(X)
X2 = scale_x.transform(X2)

Y = scale_y.transform(Y)
Y2 = scale_y.transform(Y2)

# Combined set
X3 = np.concatenate((X, X2))
Y3 = np.concatenate((Y, Y2))

# Parameterize network
N_fourier = 16
freq_init = np.pi / (np.asarray([2, 3, 4, 6, 7, 12, 24, 48, 52, 365.24]) / SCALE_MAX)
freq_init = np.concatenate((freq_init, 2 * np.pi * np.linspace(0, 1, N_fourier - len(freq_init))))
freq_init = np.concatenate((freq_init, freq_init)).reshape(1, N_fourier * 2)

# Parameterize network
offset_init = np.concatenate((
  np.ones(N_fourier) * np.pi / 2,
  np.ones(N_fourier)
)).reshape(1, N_fourier * 2)

# Neural decomposition network
keras_in = Input(shape=(1,), dtype="float32", name='keras_in')
dummy_in = Input(shape=(1,), dtype="float32", name='dummy_in')
offset = Flatten()(Embedding(
  1, N_fourier * 2, input_length=1, name='offset', embeddings_initializer='glorot_normal',
  weights = [offset_init],
  trainable=True
  )(dummy_in))
fourier_base = Dense(
  N_fourier * 2, activation='linear', use_bias=False, kernel_initializer='glorot_normal',
  weights = [freq_init],
  trainable=True,
  #kernel_constraint=min_max_norm(min_value=0.0, max_value=1.0, rate=1.0, axis=0)
  kernel_constraint=non_neg()
  )(keras_in)
fourier_base = Add()([fourier_base, offset])
fourier_base = BatchNormalization()(fourier_base)
fourier_sin = Activation(sin)(fourier_base)
fourier_cos = Activation(cos)(fourier_base)

lin_trend = Dense(
  1,
  kernel_initializer='zero',
  activation='linear',
  use_bias=True,
  )(keras_in)

quad_trend = Dense(
  1,
  kernel_initializer='zero',
  activation='linear',
  use_bias=False,
  )(keras_in)
quad_trend = Activation(quad)(quad_trend)

# sig_trend = Dense(
#   1,
#   kernel_initializer='zero',
#   activation='sigmoid',
#   use_bias=False,
#   )(keras_in)

relu_trend = Dense(
  16,
  kernel_initializer='zero',
  activation='relu',
  use_bias=True,
  )(keras_in)
relu_trend = BatchNormalization()(relu_trend)
relu_trend = Dropout(.50)(relu_trend)

joint = Concatenate()([fourier_sin, lin_trend, quad_trend, relu_trend])
# for i in range(3):
#   joint = Dense(
#   64, activation='relu',
#   kernel_regularizer=regularizers.l1(.01),
#   kernel_initializer='glorot_normal'
#   )(joint)

#joint = BatchNormalization()(joint)
output = Dense(
  1, activation="linear", kernel_regularizer=regularizers.l1(.01),
  kernel_initializer='glorot_normal'
  )(joint)

model = Model([keras_in, dummy_in], output)
model.compile(Adam(.01), loss="mse", metrics=['mae', 'mse'])

history = model.fit([X, X_dummy], Y, batch_size=64, epochs=1000)
print(np.argmin(history.history['loss']))
P1 = model.predict([X, X_dummy]).flatten()
P2 = model.predict([X2, X2_dummy]).flatten()
P3 = np.concatenate((P1, P2))

plt.plot(X3, Y3)
plt.plot(X3, P3)
plt.show()

plt.plot(X2, Y2)
plt.plot(X2, P2)
plt.show()

plt.plot(X, Y)
plt.plot(X, P1)
plt.show()

pred = model.predict(X2)
N = 168
plt.plot(X2[1:N], Y2[1:N])
plt.plot(X2[1:N], pred[1:N])
plt.show()

plot_model(model, show_shapes=True, to_file='model.png')
offset = model.layers[2].get_weights()[0]
freq = model.layers[3].get_weights()[0].reshape(1,N_fourier * 2)

# Linear
print(model.layers[11].get_weights())

# Quad
print(model.layers[8].get_weights())

# Sig
print(model.layers[13].get_weights())

imshow(freq)
imshow(offset)

model.layers[7].get_weights()
