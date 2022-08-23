from keras.layers import Input, Dense
from keras.layers import Dot, Add, Reshape, Flatten, Multiply, Activation
from keras.models import Model
from keras.backend import variable
from keras.initializers import Constant
import numpy as np
import tensorflow as tf

input_dim = 4
output_dim = 1
embed_dim = 2

w = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

model_in = Input((input_dim,), name='in', sparse=True)
#model_in_reshaped = Reshape((input_dim, 1), name='in-reshape')(model_in)
model_in_reshaped = tf.compat.v1.sparse_reshape(model_in, (input_dim, 1), name='in-reshape')

constant_tensor = Input(tensor=variable(np.ones((1,1))), name='constant', sparse=False)
embedding_weights = Dense(input_dim*embed_dim, use_bias=False, name='learnable-weights', kernel_initializer = Constant(w), trainable=False)(constant_tensor)
embedding_weights = Reshape((input_dim, embed_dim), name='reshape-weights')(embedding_weights)

#out = Multiply(name='multiply')([model_in_reshaped, embedding_weights])
out = model_in_reshaped.__mul__(embedding_weights)

# Model
model = Model([model_in, constant_tensor], out)
model.compile(loss='mse', optimizer='adam')
model.summary()

# Predict
X = np.array([[1,2,3,4],[2,4,6,8]])
print(model.predict(X, batch_size=1))
print(model.predict(X, batch_size=2))
