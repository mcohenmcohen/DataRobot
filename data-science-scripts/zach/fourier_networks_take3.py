
#nano $HOME/.keras/keras.json
#nano ~/.theanorc
#export PATH=/usr/local/cuda/bin:/usr/local/cuda/lib64:$PATH
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
#export LIBRARY_PATH=/usr/local/cuda/lib64:LIBRARY_PATH
#cd ~/Kaggle/quora/
#source ../kaggle/bin/activate
#ipython

from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam, SGD, Nadam
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Dot, Add, Concatenate, Multiply
from keras.layers import LeakyReLU, PReLU, ELU, ThresholdedReLU
from keras.activations import relu
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
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import minimize
import itertools

# TODO: Hour of day + day of week dummies (if useful)

%matplotlib osx

# Todo: try soft exponential for trend:
#https://github.com/fchollet/keras/issues/3842

# Todo:
# Input layer of LeakyReLU before sines to allow a little warping
# Explore logspace fequencies, with more higher frequency parts (low freqs seem to be easier to find)

# sin activation
def sin(x):
  return K.sin(x)
get_custom_objects().update({'sin': Activation(sin)})

#train = pd.read_csv('https://s3.amazonaws.com/datarobot_public_datasets/time_series/hyndman_turkish_electricity_demand_train.csv')
#test = pd.read_csv('https://s3.amazonaws.com/datarobot_public_datasets/time_series/hyndman_turkish_electricity_demand_test.csv')

# train = pd.read_csv('https://s3.amazonaws.com/datarobot_public_datasets/time_series/weekly_earth_co2_train.csv')
# test = pd.read_csv('https://s3.amazonaws.com/datarobot_public_datasets/time_series/weekly_earth_co2_test.csv')

train = pd.read_csv('https://s3.amazonaws.com/datarobot_public_datasets/time_series/monthly_earth_co2_train.csv')
test = pd.read_csv('https://s3.amazonaws.com/datarobot_public_datasets/time_series/monthly_earth_co2_test.csv')

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

# Test Set
# X2 = np.linspace(25, 50, 10000).reshape(-1, 1)
# Y2 = np.sin(1*X2) + np.cos(2*X2) + np.cos(3*X2) + 0.3*X2
X2 = X.max() + np.arange(test.shape[0])
Y2 = test[['y']].values

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
N_fourier = 128
freq_odd  = 2 * np.pi * (np.linspace(1, SCALE_MAX, N_fourier)/2 + 1)
freq_even = 2 * np.pi * (np.linspace(1, SCALE_MAX, N_fourier)/2)
freq_init = np.concatenate((freq_odd, freq_even)).reshape(1, N_fourier * 2)
offset_init = np.concatenate((
  np.ones(N_fourier) * np.pi / 2,
  np.ones(N_fourier) * np.pi
))

# Linear basis
time_in = Input(shape=(1,), dtype="float32", name='time_in')

#Trends
nonlin_trend = time_in
for i in range(1):
  nonlin_trend = Dense(
    8,
    kernel_initializer='zero',
    activation='relu', #'linear' for advanced
    use_bias=True,
    #kernel_regularizer=regularizers.l2(.01),
    )(nonlin_trend)
#nonlin_trend = PReLU()(nonlin_trend)
#nonlin_trend = ELU()(nonlin_trend)
#nonlin_trend = ThresholdedReLU()(nonlin_trend)
#nonlin_trend = BatchNormalization()(nonlin_trend)
#nonlin_trend = Dropout(.50)(nonlin_trend)

# Neural decomposition network
fourier = Dense(
  N_fourier * 2, activation='linear', use_bias=True, kernel_initializer='zero',
  weights = [freq_init, offset_init],
  #kernel_regularizer=regularizers.l1(.01),
  trainable=True,
  #kernel_constraint=min_max_norm(min_value=0.0, max_value=1.0, rate=1.0, axis=0)
  kernel_constraint=non_neg()
  )(time_in)
fourier = Activation(sin)(fourier)

# Combine
joint = Concatenate()([time_in, nonlin_trend, fourier])
output = Dense(
  1, activation="linear", kernel_regularizer=regularizers.l1(.01),
  kernel_initializer='glorot_normal'
  )(joint)

model = Model(time_in, output)
opt = Adam(0.001)
model.compile(opt, loss="mse", metrics=['mae', 'mse'])

history = model.fit(X, Y, batch_size=2048, epochs=20000)
# print(np.argmin(history.history['loss']))

# Fit with BFGS
# Todo: trainable weights only!!
def quick_flatten(L):
    return np.hstack([x.flatten() for x in L])

def flatten(L):
    shapes = [x.shape for x in L]
    flat = [x.flatten() for x in L]
    N = [len(x) for x in flat]
    N = N[:-1]
    N = np.cumsum(N)
    return np.hstack(flat), shapes, N

def reconstruct(flat, shapes, N):
    return [x.reshape(shapes[i]) for i,x in enumerate(np.split(flat, N))]

optimizer = model.optimizer
gradients = optimizer.get_gradients(model.total_loss, model.trainable_weights)

input_tensors = [model.inputs[0], # input data
                 model.sample_weights[0], # how much to weight each sample by
                 model.targets[0], # labels
                 K.learning_phase(), # train or test mode
]

get_gradients = K.function(inputs=input_tensors, outputs=gradients)
weights_init = [x.get_value() for x in model.trainable_weights] # CPU
#weights_init = K.get_session().run(model.trainable_weights)  # GPU
weights_init[-1] = np.zeros(weights_init[-1].shape)
weights_init[-2] = np.random.normal(size=(weights_init[-2].shape[0])).reshape(weights_init[-2].shape)
w_init, shapes, N = flatten(weights_init) # Use current model weights
w_init = np.random.normal(size=w_init.shape[0]) # Use random weights
w_init = w_init.astype(np.float32)

def set_weights(w):
  x = reconstruct(w, shapes, N)
  for i, layer in enumerate(model.trainable_weights):
    layer.set_value(x[i].astype(np.float32))

def eval_loss(w):
  set_weights(w)
  P = model.predict(X)
  return ((P - Y) ** 2).mean()

X_tmp = X.reshape(-1,1).astype(np.float32)
Y_tmp = Y.reshape(-1,1).astype(np.float32)
W = np.ones(X.shape[0]).astype(np.float32)
def eval_grad(w):
  set_weights(w)
  out = np.zeros((X.shape[0], w.shape[0]))
  inputs = [X_tmp, W, Y, 0]
  out = quick_flatten(get_gradients(inputs))
  return out.astype(np.float64)

print(eval_loss(w_init))
print(eval_grad(w_init))
res = minimize(eval_loss, w_init, method='BFGS', jac=eval_grad, options={'disp': True})
#res = minimize(eval_loss, w_init, method='L-BFGS-B', jac=eval_grad, options={'disp': True})

# Predict
P1 = model.predict(X).flatten()
P2 = model.predict(X2).flatten()
P3 = np.concatenate((P1, P2))

# Plot
plt.plot(X2, Y2)
plt.plot(X2, P2)
plt.show()

plt.plot(X3, Y3)
plt.plot(X3, P3)
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

# Fourier
freq = scale_x.inverse_transform(model.layers[3].get_weights()[0])
offset = scale_x.inverse_transform(model.layers[3].get_weights()[1])
print(np.round(freq[np.argsort(freq)], 2))
print(np.round(offset, 2))

# Linear
print(model.layers[11].get_weights())

# Quad
print(model.layers[8].get_weights())

# Sig
print(model.layers[13].get_weights())

imshow(freq)
imshow(offset)

model.layers[7].get_weights()
