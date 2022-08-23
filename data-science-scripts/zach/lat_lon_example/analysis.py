# Load libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize

def cosine_dist(a, b):
  a = normalize(a, norm='l2')
  b = normalize(b, norm='l2')
  dist = (a - b) ** 2
  dist = dist.sum(axis=1)
  dist = np.sqrt(dist)
  return dist.mean()
  
# Load data
data = pd.read_csv('~/workspace/data-science-scripts/zach/lat_lon_example/uscities.csv')
X = data[['population', 'density']].values
y = data[['lat', 'lng']].values

# Lightly preprocess X
# arcsinh is like log, but handles negatives and zeros
X = np.arcsinh(X)
X = StandardScaler().fit_transform(X)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Function to define model
# I use all zeros for the initializer to help reduce the impact of the random initialization
def build_model(output_size=2, loss='mse'):
  inputs = tf.keras.layers.Input(shape=X.shape[1], name='inputs')
  outputs = inputs
  outputs = tf.keras.layers.Dense(256, kernel_initializer='zeros', bias_initializer='zeros', activation='relu')(outputs)
  outputs = tf.keras.layers.Dense(256, kernel_initializer='zeros', bias_initializer='zeros', activation='relu')(outputs)
  outputs = tf.keras.layers.Dense(output_size, kernel_initializer='zeros', bias_initializer='zeros', activation='linear')(outputs)
  
  model = tf.keras.Model(inputs, outputs)
  model.compile(optimizer='adam', loss=loss)
  return model

EPOCHS = 10

# Method 1: independent models
lat_model = build_model(1)
lng_model = build_model(1)

np.random.seed(42)
lat_model.fit(X_train, y_train[:,0], batch_size=32, epochs=EPOCHS, shuffle=True)
np.random.seed(42)
lng_model.fit(X_train, y_train[:,1], batch_size=32, epochs=EPOCHS, shuffle=True)

pred_test = np.zeros(y_test.shape)
pred_test[:,0] = lat_model.predict(X_test).ravel()  # 4822.101367905647
pred_test[:,1] = lng_model.predict(X_test).ravel()  # 0.3870690611361487
print(mean_squared_error(y_test, pred_test))
print(cosine_dist(y_test, pred_test))
pd.DataFrame(pred_test).to_csv('~/workspace/data-science-scripts/zach/lat_lon_example/2_model_preds.csv', index=False)

# Method 2: 1 model with independent loss
both_model = build_model(2)
np.random.seed(42)
both_model.fit(X_train, y_train, batch_size=32, epochs=EPOCHS, shuffle=True)

pred_test = both_model.predict(X_test)
print(mean_squared_error(y_test, pred_test))  # 4822.09783484137
print(cosine_dist(y_test, pred_test))  # 0.3882481766857536
pd.DataFrame(pred_test).to_csv('~/workspace/data-science-scripts/zach/lat_lon_example/1_model_preds.csv', index=False)

# Method 3: 1 model with cosine loss
cos_model = build_model(2, 'CosineSimilarity')
np.random.seed(42)
cos_model.fit(X_train, y_train, batch_size=32, epochs=EPOCHS, shuffle=True)

pred_test = cos_model.predict(X_test)
print(mean_squared_error(y_test, pred_test))  # 5203.022940294414
print(cosine_dist(y_test, pred_test))  # 0.053871665424494265
pd.DataFrame(pred_test).to_csv('~/workspace/data-science-scripts/zach/lat_lon_example/1_cos_model_preds.csv', index=False)
