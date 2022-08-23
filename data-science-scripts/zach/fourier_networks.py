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

# Todo: min/max scale time

class FourierLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(FourierLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.frequency = self.add_weight(
          name='frequency',
          shape=(input_shape[1], self.output_dim),
          initializer='he_normal',
          trainable=True)
        self.phase_shift = self.add_weight(
          name='phase_shift',
          shape=(1, self.output_dim),
          initializer='he_uniform',
          trainable=True)
        super(FourierLayer, self).build(input_shape)

    def call(self, x):
        out = K.dot(x, self.frequency) +  K.flatten(self.phase_shift)
        return K.sin(out)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

# Train set
X = np.linspace(0, 1, 128)
Y = np.sin(4.25*np.pi*X) + np.sin(8.5*np.pi*X) + 5*X

# Test Set
X2 = np.linspace(1, 3, 256)
Y2 = np.sin(4.25*np.pi*X2) + np.sin(8.5*np.pi*X2) + 5*X2

# Scale
scale_x = MinMaxScaler((0,1))
scale_x.fit(X)

scale_y = MinMaxScaler((0,10))
scale_y.fit(Y)

X = scale_x.transform(X)
X2 = scale_x.transform(X2)

Y = scale_x.transform(Y)
Y2 = scale_x.transform(Y2)

# Combined set
X3 = np.concatenate((X, X2))
Y3 = np.concatenate((Y, Y2))

# Sine part
keras_in = Input(shape=(1,), dtype="float32", name='users_in')
sine_network = FourierLayer(8)(keras_in)

# Non-periodic part
regular_network = keras_in
for i in range(1):
  regular_network = Dense(
    1,
    activation='linear',
    use_bias=True,
    )(regular_network)
  #regular_network = Dropout(.5)(regular_network)
  #regular_network = BatchNormalization()(regular_network)

joint_network = Concatenate()([sine_network, regular_network])
model_output = Dense(
  1, activation="linear", kernel_regularizer=regularizers.l1(0.01)
  )(joint_network)

model = Model(keras_in, model_output)
model.compile(Adam(0.01), loss="mse", metrics=['mse'])

history = model.fit(X, Y, batch_size=128, epochs=2500)
print(np.argmin(history.history['loss']))
P1 = model.predict(X).flatten()
P2 = model.predict(X2).flatten()
P3 = np.concatenate((P1, P2))

plt.plot(X2, Y2)
plt.plot(X2, P2)
plt.show()

plt.plot(X, Y)
plt.plot(X, P1)
plt.show()

plt.plot(X3, Y3)
plt.plot(X3, P3)
plt.show()

pred = model.predict(X2)
N = 168
plt.plot(X2[1:N], Y2[1:N])
plt.plot(X2[1:N], pred[1:N])
plt.show()

