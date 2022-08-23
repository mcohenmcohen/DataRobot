# https://tfhub.dev/google/elmo/2
# TF 1.4.1 may be needed on the GPU machine
from keras import layers
from keras.layers import Layer
from keras.models import Model
import keras.backend as K
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from time import time
from pandas import read_csv
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import normalize

class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable = False
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable, name="{}_module".format(self.name))
        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                      as_dict=True,
                      signature='default',
                      )['default']
        return result

    def compute_mask(self, inputs, mask=None):
      return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
      return (input_shape[0], self.dimensions)


input_text = layers.Input(shape=(1,), dtype=tf.string)
embedding = ElmoEmbeddingLayer()(input_text)

model = Model(inputs=[input_text], outputs=embedding)
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

p1 = model.predict(np.array(["I don't like that"]))
p2 = model.predict(np.array(["I do not like that"]))
print((normalize(p1) * normalize(p2)).sum())

p1 = model.predict(np.array(["I took an Uber"]))
p2 = model.predict(np.array(["I took an Uberx"]))
print((normalize(p1) * normalize(p2)).sum())

p1 = model.predict(np.array(["I took an Uber"]))
p2 = model.predict(np.array(["I took an Uberxl"]))
print((normalize(p1) * normalize(p2)).sum())

p1 = model.predict(np.array(["I took an Uberx"]))
p2 = model.predict(np.array(["I took an Uberxl"]))
print((normalize(p1) * normalize(p2)).sum())


t1 = time()
p = model.predict(np.array([
  'This is my text',
  'Zach',
  'zach',
  "the cat is on the mat",
  "dogs are in the fog"
]))
t2 = time()
print(t2-t1)
print(p.shape)

p1 = model.predict(np.array(['I hate uber!']))
p2 = model.predict(np.array(['I love uber!']))

print((p1 * p2).sum())
print((normalize(p1) * normalize(p2)).sum())

p1 = model.predict(np.array(['おはようございます ']))
p2 = model.predict(np.array(['こんにちは']))

print((p1 * p2).sum())
print((normalize(p1) * normalize(p2)).sum())

# Lamborgini vs Ferrari - Japanese
p1 = model.predict(np.array(['ランボルギーニ']))
p2 = model.predict(np.array(['フェラーリ']))
print((normalize(p1) * normalize(p2)).sum())

# Lamborgini vs Ferrari - English
p1 = model.predict(np.array(['lamborghini']))
p2 = model.predict(np.array(['ferrari']))
print((normalize(p1) * normalize(p2)).sum())

# Emojis
p1 = model.predict(np.array(['🙁']))
p2 = model.predict(np.array(['😊']))
print((normalize(p1) * normalize(p2)).sum())

# Emojis
p1 = model.predict(np.array(['🙁']))
p2 = model.predict(np.array(['😃']))
print((normalize(p1) * normalize(p2)).sum())

# Emojis
p1 = model.predict(np.array(['😃']))
p2 = model.predict(np.array(['😊']))
print((normalize(p1) * normalize(p2)).sum())

# 10k DB
data = read_csv('https://s3.amazonaws.com/datarobot_public_datasets/sas/10k_diabetes.csv')
string_data = data['diag_1_desc'].astype('str').values
for i in range(string_data.shape[0]):
  string_data[i] = str(string_data[i])
t1 = time()
diag_I enc = model.predict(string_data, batch_size=4096)
t2 = time()
print(t2-t1)

out_enc = pd.DataFrame(normalize(diag_enc))
out_enc['readmitted'] = data['readmitted']
out_enc.to_csv('10k_db_encoded_norm.csv', index=False)

out_raw = data[['readmitted', 'diag_1_desc']]
out_raw.to_csv('10k_db_raw.csv', index=False)

# Yelp polarity
data = read_csv('https://s3.amazonaws.com/datarobot_data_science/test_data/yelp_review_polarity_full.csv')
string_data = data['text'].astype('str').values
for i in range(string_data.shape[0]):
  string_data[i] = str(string_data[i])
t1 = tstring_pred = np.zeros((data.shape[0], 1024))
chunk_size = 128
t1 = time()
for i in tqdm(range(0, len(string_data), chunk_size)):
  string_pred[i:(i+chunk_size),:] = model.predict(string_data[i:(i+chunk_size)])
print(t2-t1)  # 1411617.584883213 seconds or 17 days

out_enc = pd.DataFrame(normalize(string_pred))
out_enc['class_id'] = data['class_id']
out_enc.to_csv('yelp_enc_norm.csv', index=False)

# Airbnb Rating
data = read_csv('https://s3.amazonaws.com/datarobot_data_science/test_data/Airbnb-combined.csv')
string_data = data['review'].astype('str').values
for i in range(string_data.shape[0]):
  string_data[i] = str(string_data[i])
t1 = time()
enc = model.predict(string_data, batch_size=512)
t2 = time()
print(t2-t1)

out_enc = pd.DataFrame(normalize(enc))
out_enc['rating'] = data['rating']
out_enc.to_csv('airbnb_enc_norm.csv', index=False)
