import numpy as np
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Input, Dense, Activation
from keras.models import Model
import keras.backend as K
from tensorflow import distributions
np.random.seed(42)

# Some Data
nrows = 1000
ncols = 20
X = np.random.rand(nrows, ncols) - 0.5
CF = np.random.rand(ncols, 1)
y = np.sign(X.dot(CF))
y[np.where(y == -1)] = 0

# Test data
X_test = np.random.rand(5000, ncols) - 0.5
y_test = np.sign(X_test.dot(CF))
y_test[np.where(y_test == -1)] = 0

# Links
# http://m-hikari.com/ams/ams-2014/ams-85-88-2014/epureAMS85-88-2014.pdf
def approx_probit1(x):
  return 2 ** (-22 ** (1 - 41 ** (x / 10)))

def probit(x):
    normal = distributions.Normal(loc=0., scale=1.)
    return normal.cdf(x)

def cloglog(x):
    return -(K.exp(-K.exp(x))-1)

# logit model
input = Input(shape=(ncols,))
output = Dense(1, activation='sigmoid', kernel_initializer='zeros')(input)
logit_model = Model(input, output)
logit_model.compile(loss='binary_crossentropy', optimizer='adam')
logit_model.fit(X, y, epochs=1000)

# probit model
input = Input(shape=(ncols,))
output = Dense(1, activation=probit, kernel_initializer='zeros')(input)
probit_model = Model(input, output)
probit_model.compile(loss='binary_crossentropy', optimizer='adam')
probit_model.fit(X, y, epochs=1000)

# cloglog model
input = Input(shape=(ncols,))
output = Dense(1, activation=cloglog, kernel_initializer='zeros')(input)
cloglog_model = Model(input, output)
cloglog_model.compile(loss='binary_crossentropy', optimizer='adam')
cloglog_model.fit(X, y, epochs=1000)

# Eval models
print(logit_model.evaluate(X_test, y_test, verbose=False))
print(probit_model.evaluate(X_test, y_test, verbose=False))
print(cloglog_model.evaluate(X_test, y_test, verbose=False))
