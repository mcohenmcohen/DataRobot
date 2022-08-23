import wordfreq
import pandas as pd
import numpy as np
import numpy as npy
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import normalize
from scipy.fft import fft
from scipy.signal import tukey

# Tokenize one text by its word frequencie
def frequnecy_tokenize(x, lang='en'):
  tokens = wordfreq.tokenize(x, lang=lang)
  return [wordfreq.zipf_frequency(x, lang=lang, wordlist='large') for x in tokens]

# Tokenize many texts by their word frequencies and truncate/pad them to the same length
def frequnecy_vectorize(text_col, lang='en'):
  out = text_col.apply(frequnecy_tokenize)
  out = out.apply(lambda x: 1/(np.array(x) + 1))
  out = pad_sequences(out, maxlen=200, dtype='float32', padding='post', truncating='post')
  out = normalize(out, norm='l2', copy=False)
  out = np.abs(fft(out, norm='ortho'))
  return out

# Load data
train_df = pd.read_csv('output/data_with_folds.csv')
test_df = pd.read_csv('input/test.csv')

# Vectorize the training data
freq_vec = frequnecy_vectorize(train_df['excerpt'])
for i in range(200):
  print(np.corrcoef(freq_vec[:,i], train_df['target'].values)[0][1])
train_df_x = pd.DataFrame(freq_vec).reset_index(drop=True)

# Save for DataRobot
se = train_df['standard_error'].values
se[np.where(se==0)] = 1
train_df['weight'] = 1 / se

cols = ['fold', 'weight', 'target']
out = pd.concat([train_df[cols].reset_index(drop=True), train_df_x], axis=1)
out.to_csv('freq_encoded.csv', index=False)


#https://stackoverflow.com/questions/52690632/analyzing-seasonality-of-google-trend-time-series-using-fft
a_gtrend_orig = freq_vec[0,:]
t_gtrend_orig = np.linspace( 0, len(a_gtrend_orig)/12, len(a_gtrend_orig), endpoint=False )
a_gtrend_windowed = (a_gtrend_orig-np.median( a_gtrend_orig ))*tukey( len(a_gtrend_orig) )
