#https://arxiv.org/pdf/1708.05123.pdf
#Deep & Cross Network for Ad Click Predictions

from keras.layers import Input, Dense
from keras.layers import Dot, Add, Reshape, Flatten
from keras.models import Model

def make_cross(network_input, cross_input, shape):
    """deep cross"""
    input_reshaped = Reshape((-1, 1))(network_input)
    cross_input_reshaped = Reshape((-1, 1))(cross_input)
    cross_input_reshaped = Dense(shape)(cross_input_reshaped)
    cross_input_reshaped = Dense(1)(cross_input_reshaped)
    cross_input_reshaped = Reshape((-1, 1))(cross_input_reshaped)

    cross = Dot(axes=2)([input_reshaped, cross_input_reshaped])
    cross = Reshape((-1, 1))(cross)
    out = Dense(shape)(cross)
    out = Dense(1)(out)
    out = Flatten()(out)
    out = Add()([cross_input, out])
    return out

N = 10
input = Input((25,))
output = input
for _ in range(1):
    output = make_cross(input, output, N)

model = Model(input, output)
model.summary()
