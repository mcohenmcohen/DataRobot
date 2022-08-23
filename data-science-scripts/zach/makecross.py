from keras.layers import Input, Dense
from keras.layers import Dot, Add, Reshape, Flatten, TimeDistributed, Conv1D, BatchNormalization
from keras.models import Model
import keras.backend as K
import numpy as np

def make_cross(network_input, cross_input, shape):

    # Dot product of inputs
    input_reshaped = Reshape((1, shape))(network_input)
    cross_input_reshaped = Reshape((1, shape))(cross_input)
    cross = Dot(axes=1)([input_reshaped, cross_input_reshaped])

    # Collapse from NxN to Nx1
    out = cross
    out = Flatten()(out)
    out = Dense(shape)(out)

    # Add to the previous layer
    out = Add()([cross_input, out])
    return(out)

N = 10
input = Input((N,))
out = input
for _ in range(1):
    out = make_cross(input, out, N)

model = Model(input, out)
model.summary()

# debug
network_input = input
cross_input = out
shape = N
