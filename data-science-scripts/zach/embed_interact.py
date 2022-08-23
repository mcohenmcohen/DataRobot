from keras.layers import Input, Dense
import keras.backend as K

# Settings
cols = 100
embed = 64

# Input layer
input_layer = Input((cols,embed), name='in', sparse=False)

# Interactions
square_of_sum = K.square(K.sum(input_layer, axis=1, keepdims=False))
sum_of_square = K.sum(input_layer * input_layer, axis=1, keepdims=False)
cross_term = 0.5 * (square_of_sum - sum_of_square)






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
