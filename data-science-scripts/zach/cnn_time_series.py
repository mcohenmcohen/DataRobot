from __future__ import print_function
import itertools
import gc
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.engine import Layer, InputSpec
from keras.layers import Input, Embedding, Dense, Dropout, Reshape
from keras.layers import Merge, BatchNormalization, TimeDistributed, Lambda, Flatten
from keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten
from keras.layers.merge import Concatenate, Dot, Multiply
from keras.layers.recurrent import GRU, LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import Sequential
from keras.regularizers import l2
from keras import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, log_loss
from tqdm import tqdm
from sklearn.preprocessing import normalize
from keras.optimizers import Adam
from keras.constraints import non_neg, unit_norm
from sklearn.preprocessing import normalize
from keras.layers.convolutional import Conv1D, Conv2D
from keras.layers.convolutional import MaxPooling1D, MaxPooling2D
from os import system
from keras.utils import plot_model

# https://gist.github.com/jkleint/1d878d0401b28b281eb75016ed29f2ee

dat = np.genfromtxt('/Users/zachary/datasets/SPY.csv', delimiter=',', skip_header=1)

X = dat[:,0:5]
y = dat[:,6]

history = 10
X_new = np.zeros((X.shape[0], history, X.shape[1]))
for i in range(X.shape[0]-history+1):
  start = i
  stop = i+history
  X_new[stop-1,:,:] = X[start:stop,].reshape((1, history, X.shape[1]))

# Check that we have the first row of history in the first row our our new data
assert np.all(X_new[history-1,0,:] == X[0,])

# Check that we have the last row of history in the last row our our new data
assert np.all(X_new[-1,-1,:] == X[-1,])

# Check that the last y is not in our data
assert X[-1,3] != y[-1,]

# Check that the second to last y is not in our data
assert X[-1,3] == y[-2,]

# Define model
# Todo: add second input for raw data from today
nb_filter = 16
conv_filters = []
inp = Input(shape=(history,5))
for n in [1,2,3]:
    model = inp
    model = BatchNormalization()(model)
    model = Convolution1D(16, kernel_size=n, activation='relu')(model)
    model = MaxPooling1D()(model)
    #model = Convolution1D(16, kernel_size=n, activation='relu')(model)
    #model = MaxPooling1D()(model)
    model = Flatten()(model)
    conv_filters.append(model)

# Todo: Add L2
mlp = Concatenate()(conv_filters)
for i in range(3):
  mlp = Dense(32, activation='relu')(mlp)
  #mlp = Dropout(.10)(mlp)
  #mlp = BatchNormalization()(mlp)
mlp = Dense(1, activation="linear")(mlp)

model = Model(inp, mlp)
model.compile(Adam(0.005), loss="mse")

# Todo: time based validation
model.fit(
  X_new[history:,:], y[history:],
  validation_split=.20,
  epochs=1000,
  verbose=1,
  shuffle=True,
  batch_size=128
  )
  
plot_model(model, to_file='model.png')

