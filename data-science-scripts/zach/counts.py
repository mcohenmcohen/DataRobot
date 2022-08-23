import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

doc = pd.read_csv('~/datasets/gibberish.txt', header=None, encoding='utf-8')
vec = CountVectorizer(lowercase=False)
vec.fit_transform(doc.iloc[:,0])
idx = np.argmax(np.fromiter(vec.vocabulary_.values(), dtype='int32'))
print(list(vec.vocabulary_.items())[idx])
