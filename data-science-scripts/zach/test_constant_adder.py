
#################################################################
# Dense method — trainable
#################################################################

import numpy as np
from keras.layers import Input, Dense
from keras.initializers import Constant
from keras.models import Model

inputs = Input(shape=(1,))
output = Dense(1, kernel_initializer=Constant(value=1), bias_initializer=Constant(value=5), trainable=False)(inputs)
model = Model(inputs, output)
model.compile(loss='mse', optimizer='adam')
X = np.array([1,2,3,4,5,6,7,8,9,10])
model.predict(X)

inputs = Input(shape=(2,))
output = Dense(2, kernel_initializer=Constant(value=[1,1]), bias_initializer=Constant(value=[5,10]), trainable=False)(inputs)
model = Model(inputs, output)
model.compile(loss='mse', optimizer='adam')
X = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(5, 2)
model.predict(X)

#################################################################
# SO method
#################################################################
# https://stackoverflow.com/questions/52413371/save-load-a-keras-model-with-constants

import numpy as np
from keras.layers import Input, Lambda
from keras.models import Model, load_model

def add_five(a):
    return a + 5

inputs = Input(shape=(1,))
output = Lambda(add_five)(inputs)

model = Model(inputs, output)
model.compile(loss='mse', optimizer='adam')

X = np.array([1,2,3,4,5,6,7,8,9,10])
model.predict(X)

# ... the same as above (just change the function name and input shape)
def add_constants(a):
    return a + [5, 10]
inputs = Input(shape=(2,))
output = Lambda(add_constants)(inputs)
model = Model(inputs, output)
model.compile(loss='mse', optimizer='adam')
X = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(5, 2)
model.predict(X)

#################################################################
# Use constant
#################################################################

# Define model 1
from keras.layers import Input, Add, Multiply
from keras.backend import variable
from keras.models import Model, clone_model
import numpy as np
from keras.models import load_model

inputs = Input(shape=(1,))
add_in = Input(tensor=variable([[5]]), name='add')
output = Add()([inputs, add_in])

model = Model([inputs, add_in], output)
model.compile(loss='mse', optimizer='adam')

X = np.array([1,2,3,4,5,6,7,8,9,10])
model.predict(X)

# Save the model, delete it, and reload
p = 'k_model.hdf5'
model.save(p)
del model
model2 = load_model(p)
model2.predict(X)

model2.predict([X, X])

inputs = Input(shape=(2,))
output = inputs
std_input = Input(tensor=variable([[10, 100]]), name='stds')
mean_input = Input(tensor=variable([[5, 50]]), name='means')
output = Multiply()([output, std_input])
output = Add()([output, mean_input])

model = Model([inputs, std_input, mean_input], output)
model.compile(loss='mse', optimizer='adam')


a = np.array([[1,2,3,4,5,6,7,8,9,10], [1,2,3,4,5,6,7,8,9,10]]).transpose()
model.fit(a, a)
model.predict(a)
