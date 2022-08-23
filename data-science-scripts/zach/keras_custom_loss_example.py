# Imports
import tensorflow as tf
import numpy as np

# Make sample data
rows = 100
cols = 2
X = np.random.rand(rows, cols)
y = np.random.rand(rows, 1)

# Define the keras model
inputs = tf.keras.Input(shape=(X.shape[1],))
outputs = tf.keras.layers.Dense(1)(inputs)
estimator = tf.keras.Model(inputs, outputs)

# Define and test custom loss
def custom_loss_fn(y_true, y_pred):
   loss = tf.keras.backend.square(y_pred - y_true)
   loss = tf.keras.backend.sum(loss, axis=0)
   return loss

assert custom_loss_fn(y, y).numpy() == np.zeros(y.shape[1])
assert (custom_loss_fn(X, X).numpy() == np.zeros(X.shape[1])).all()

# Compile the model with the custom loss function
estimator.compile(loss=custom_loss_fn, optimizer='adam')

# Test that the model can predict
y_pred = estimator.predict(X)
assert y_pred.shape == y.shape

# Test that the model can fit
estimator.fit(X, y)
