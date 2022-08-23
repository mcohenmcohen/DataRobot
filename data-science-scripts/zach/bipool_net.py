from keras.layers import Input, Dense
from keras.layers import Dot, Add, Reshape, Flatten, Multiply
from keras.models import Model

# References
# https://deepctr-doc.readthedocs.io/en/latest/Features.html
# https://arxiv.org/pdf/1708.05027.pdf
# https://arxiv.org/pdf/1902.09096.pdf

# Settings
input_dim = 100
output_dim = 1
embed_dim = 10

# Input layer
input = Input((input_dim,), name='in')
out = input

# Shared "embedding" space
out = Dense(embed_dim, name='embed')(out)

# Elementwise product
out = Multiply(name='product')([out, out])

# MLP layers
for i in range(3):
    out = Dense(embed_dim, name='mlp'+str(i))(out)

# Output layer
out = Dense(output_dim, name='output')(out)

# Linear part
linear = Dense(output_dim, name='linear', use_bias=False)(input)
out = Add(name='add')([out, linear])

# Model
model = Model(input, out)
model.summary()
