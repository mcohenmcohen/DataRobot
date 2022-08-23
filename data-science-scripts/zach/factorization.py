from keras.models import Model
from keras.layers import Input, Dot, Flatten, Concatenate, GlobalAveragePooling1D
from keras.layers.core import Dense, Reshape
from keras.layers.embeddings import Embedding
from keras.utils import plot_model
import keras.backend as K

# https://stackoverflow.com/questions/39510809/mean-or-max-pooling-with-masking-support-in-keras
class GlobalAveragePooling1DMasked(GlobalAveragePooling1D):
    def call(self, x, mask=None):
        if mask != None:
            return K.sum(x, axis=1) / K.sum(mask, axis=1)
        else:
            return super().call(x)

n_factors = 5
n_columns = 2000

# Dense
input = Input(shape=(n_columns,))
factors = Dense(n_factors)(input)
factors = Reshape((n_factors, 1))(factors)
interactions = Dot(axes=2)([factors, factors])
interactions = Flatten()(interactions)

merged = Concatenate()([input, interactions])
output = Dense(1)(merged)
model = Model(input, output)
model.compile(loss='mse', optimizer='adam')
model.summary()
plot_model(model, to_file='/Users/zachary/workspace/data-science-scripts/zach/dense_factorization_model.png', show_shapes=True, show_layer_names=False)

# Sparse
max_non_zero_cols = 7
input = Input(shape=(max_non_zero_cols,))

lin_embed = Embedding(n_columns, 1, input_length=max_non_zero_cols, mask_zero=False)(input)

quad_embed = Embedding(n_columns, n_factors, input_length=max_non_zero_cols, mask_zero=False)(input)
quad_embed = Dot(axes=2)([quad_embed, quad_embed])
quad_embed = Reshape((max_non_zero_cols**2, 1))(quad_embed)

merged = Concatenate(axis=1)([lin_embed, quad_embed])
merged = GlobalAveragePooling1D()(merged)  # Doesn't support mask.  Zeros will bring down average.  Can't do sum layer.

output = Dense(1)(merged)
model = Model(input, output)
model.compile(loss='mse', optimizer='adam')
model.summary()
plot_model(model, to_file='/Users/zachary/workspace/data-science-scripts/zach/sparse_factorization_model.png', show_shapes=True, show_layer_names=False)
