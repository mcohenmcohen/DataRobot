from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Input, Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Dot, Add, Concatenate, Multiply
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras import regularizers
import pandas as pd
from keras.constraints import non_neg
from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import RandomNormal
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class FourierLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(FourierLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.frequency = self.add_weight(
          name='frequency',
          shape=(input_shape[1], self.output_dim),
          initializer='glorot_normal',
          trainable=True)
        self.phase_shift = self.add_weight(
          name='phase_shift',
          shape=(1, self.output_dim),
          initializer='glorot_normal',
          trainable=True)
        super(FourierLayer, self).build(input_shape)

    def call(self, x):
        out = K.dot(x, self.frequency) +  K.flatten(self.phase_shift)
        return K.sin(out)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

# train = pd.read_csv('https://s3.amazonaws.com/datarobot_public_datasets/time_series/hyndman_turkish_electricity_demand_train.csv')
# test = pd.read_csv('https://s3.amazonaws.com/datarobot_public_datasets/time_series/hyndman_turkish_electricity_demand_test.csv')

def fun(x):
  return np.sin(1*np.pi*x) + np.cos(2*np.pi*x) + np.cos(3*np.pi*x) + np.random.normal(0,.1, N)

N = 2000
X = np.random.uniform(0,10,N)
X = X[np.argsort(X)]
Y = fun(X)

X2 = np.random.uniform(10,20,N)
X2 = X2[np.argsort(X2)]
Y2 = fun(X2)

# Scale
scale_x = MinMaxScaler((0,1000))
scale_x.fit(X)

scale_y = MinMaxScaler((0,10))
scale_y.fit(Y)

X = scale_x.transform(X)
X2 = scale_x.transform(X2)

Y = scale_y.transform(Y)
Y2 = scale_y.transform(Y2)

# Sine part
keras_in = Input(shape=(1,), dtype="float32", name='users_in')
sine_network = FourierLayer(64)(keras_in)

# sine_network = Dropout(.10)(sine_network)
# sine_network = Dropout(dropout_prob)(sine_network)
# sine_network = BatchNormalization()(sine_network)

# Non-periodic part
regular_network = keras_in
for i in range(1):
  regular_network = Dense(
    2,
    activation='linear',
    use_bias=False
    )(regular_network)
  regular_network = Dropout(.5)(regular_network)
  regular_network = BatchNormalization()(regular_network)

joint_network = Concatenate()([sine_network, regular_network])
model_output = Dense(
  1, activation="linear", kernel_regularizer=regularizers.l1(0.01)
  )(joint_network)

model = Model(keras_in, model_output)
model.compile(Adam(0.001), loss="mse", metrics=['mse'])

model.fit(X, Y, batch_size=256, epochs=500)
P1 = model.predict(X).flatten()
P2 = model.predict(X2).flatten()
P3 = np.concatenate((P1, P2))

plt.plot(X2, Y2)
plt.plot(X2, P2)
plt.show()

plt.plot(X, Y)
plt.plot(X, P1)
plt.show()
