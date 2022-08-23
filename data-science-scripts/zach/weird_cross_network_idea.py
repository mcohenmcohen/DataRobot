from keras.layers import Input, Dense
from keras.layers import Dot, Add, Reshape, Flatten, Multiply
from keras.models import Model

# Settings
input_dim = 100
output_dim = 1
reduced_dim = 10

# Input layer
input = Input((input_dim,), name='in')
out = input

# Reduce input dimensionality to something more manageable
# Maybe we could initialize the weights of this layer with PCA or something
out = Dense(reduced_dim, name='embed')(out)

# Take the cross product of the reduce
out = Reshape((1, reduced_dim))(out)
out = Dot(axes=1, name='cross')([out, out])

# MLP layers
for i in range(3):
    out = Dense(reduced_dim, name='mlp'+str(i))(out)
out = Flatten()(out)
out = Dense(output_dim, name='output')(out)

# Linear part
linear = Dense(output_dim, name='linear', use_bias=False)(input)
out = Add(name='add')([out, linear])

# Model
model = Model(input, out)
model.summary()
