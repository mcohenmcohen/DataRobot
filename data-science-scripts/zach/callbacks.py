from __future__ import division, print_function, absolute_import

from io import BytesIO

from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense, Activation, Add, Input, BatchNormalization
from keras.initializers import constant
from keras.models import Model
import keras.backend as K
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import requests
from keras.optimizers import Adam, SGD
from numpy.testing import assert_almost_equal
from scipy.sparse import load_npz, csr_matrix
from six.moves import cPickle as pickle
from sklearn.metrics import mean_squared_error

import statsmodels.api as sm
lowess = sm.nonparametric.lowess

def request_raw_bytes(x):
    return BytesIO(requests.get(x, stream=True).raw.read())


X_url = 'https://s3.amazonaws.com/datarobot_public_datasets/fit_X.npz'
y_url = 'https://s3.amazonaws.com/datarobot_public_datasets/fit_y_logged.npy'
X = load_npz(request_raw_bytes(X_url))
y = np.load(request_raw_bytes(y_url))
y = np.log1p(np.exp(y))

rng = np.random.RandomState(42)
indices = rng.permutation(X.shape[0])
split = int(X.shape[0] * 0.80)
train_idx, test_idx = indices[:split], indices[split:]

class LR_Finder(Callback):
    """
    Modified from http://puzzlemusa.com/2018/05/14/learning-rate-finder-using-keras/
    """

    def __init__(self, start_lr=1e-5, end_lr=10, step_size=None, beta=.98):
        super().__init__()

        self.start_lr = start_lr
        self.end_lr = end_lr
        self.step_size = step_size
        self.beta = beta
        self.lr_mult = (end_lr/start_lr)**(1/step_size)

    def on_train_begin(self, logs=None):
        self.best_loss = 1e9
        self.avg_loss = 0
        self.losses, self.smoothed_losses, self.lrs, self.iterations = [], [], [], []
        self.iteration = 0
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.start_lr)

    def on_batch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        self.iteration += 1

        self.avg_loss = self.beta * self.avg_loss + (1 - self.beta) * loss
        smoothed_loss = self.avg_loss / (1 - self.beta**self.iteration)

        # Check if the loss is not exploding
        if self.iteration>1 and smoothed_loss > self.best_loss * 4:
            self.model.stop_training = True
            return

        if smoothed_loss < self.best_loss or self.iteration==1:
            self.best_loss = smoothed_loss

        lr = self.start_lr * (self.lr_mult**self.iteration)

        self.losses.append(loss)
        self.smoothed_losses.append(smoothed_loss)
        self.lrs.append(lr)
        self.iterations.append(self.iteration)

        K.set_value(self.model.optimizer.lr, lr)

# Define model
def build_model():
    model_in = Input(shape=(X.shape[1],), dtype='float32', sparse=True)
    out = model_in
    out = Dense(192, activation='relu')(out)
    out = Dense(64, activation='relu')(out)
    out = Dense(64, activation='relu')(out)
    out = Dense(1, kernel_initializer=constant(y[train_idx].mean()))(out)
    residual = Dense(1, activation='linear', use_bias=False)(model_in)
    out = Add()([out, residual])
    model = Model(model_in, out)
    model.summary()
    return model

# Find learning rate
model = build_model()
model.compile(loss='mse', optimizer=Adam(0.01))
bs = 1024
train_idx_lrf = train_idx
max_iter = 100
if len(train_idx) > max_iter*bs:
    train_idx_lrf = train_idx[1:max_iter*bs]

# Fit LR find model
lrf = LR_Finder(start_lr=1e-6, end_lr=1, step_size=np.ceil(len(train_idx_lrf)/bs))
model.fit(
    X[train_idx_lrf, :], y[train_idx_lrf],
    batch_size=bs, epochs=1, verbose=True,
    callbacks=[lrf])

# Smooth LR
order = np.argsort(lrf.lrs)[::-1]
order_loss = np.array(lrf.losses)[order]
order_smoothed_loss = np.array(lrf.smoothed_losses)[order]
ordered_lr = np.array(lrf.lrs)[order]
ordered_log_lr = np.log(ordered_lr)

ordered_lowess_loss = lowess(
    order_loss, ordered_log_lr,
    frac=.15, it=3, return_sorted=True, is_sorted=False
    )

#Gradient
ordered_smoothed_grad = np.gradient(ordered_lowess_loss[:,1], ordered_lowess_loss[:,0])
best_idx = np.argmin(ordered_smoothed_grad)
best_lr = np.exp(ordered_lowess_loss[:,0][best_idx])
print(best_lr)

# Plot grad
plt.figure(figsize=(10,6))
plt.scatter(np.arange(len(ordered_smoothed_grad)), ordered_smoothed_grad)
plt.show()

# Plot LR
plt.figure(figsize=(10,6))
plt.scatter(ordered_lr, order_loss, facecolors='none', edgecolor='darkblue')
plt.plot(np.exp(ordered_lowess_loss[:,0]), ordered_lowess_loss[:,1], color='black')
plt.scatter(np.exp(ordered_lowess_loss[best_idx,0]), ordered_lowess_loss[best_idx,1], facecolors = 'red', edgecolor = 'red')
plt.xscale('log')
plt.show()

# Refit with optimal learning rate
model = build_model()
model.compile(loss='mse', optimizer=Adam(best_lr))
model.fit(
    X[train_idx, :], y[train_idx],
    batch_size=bs, epochs=1, verbose=True)

# Evaluate mpdel
pred = model.predict(X[test_idx, :])
err = np.sqrt(mean_squared_error(pred, y[test_idx]))
print(err)
