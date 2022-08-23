import pandas as pd

adult = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data')
adult.columns = ['age', 'workclass', 'fnlwgt', 'edu', 'edu_num', 'marital_status',
				 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
				 'hours_per_week', 'native_country', 'income']
adult["sex"] = adult["sex"].astype('category').cat.codes
adult["workclass"] = adult["workclass"].astype('category').cat.codes
adult["marital_status"] = adult["marital_status"].astype('category').cat.codes
adult["race"] = adult["race"].astype('category').cat.codes
adult["occupation"] = adult["occupation"].astype('category').cat.codes
adult["native_country"] = adult["native_country"].astype('category').cat.codes

target_bin = adult[['sex']].values.astype('float32')
target_num = adult[['age']].values.astype('float32')
X = adult[['workclass', 'edu_num', 'marital_status', 'occupation', 'race', 'capital_gain',
		   'capital_loss', 'hours_per_week', 'native_country']].values.astype('float32')

from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model

input = Input(shape=(X.shape[1],), name='Input')
hidden = Dense(64, name='Shared-Hidden-Layer', activation='relu')(input)
hidden = BatchNormalization()(hidden)
out_bin = Dense(1, name='Output-Bin', activation='sigmoid')(hidden)
out_num = Dense(1, name='Output-Num', activation='linear')(hidden)

model = Model(input, [out_bin, out_num])
model.compile(optimizer=Adam(0.10), loss=['binary_crossentropy', 'mean_squared_error'])
model.summary()
plot_model(model, to_file='model.png')

model.fit(X, [target_bin, target_num], validation_split=.20, epochs=100, batch_size=2048)
model.predict(X)
