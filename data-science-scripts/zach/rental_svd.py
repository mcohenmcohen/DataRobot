import gc
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA

dat = pd.read_csv('https://s3.amazonaws.com/datarobot_public_datasets/rental_train.csv')

# Encode text as sparse and drop from DF
text_vars = [
  'description',
  'photos',
  'street_address',
  'features'
]
text = {}
for var in text_vars:
    print(var)
    x = dat[var].fillna('')
    tv = TfidfVectorizer(
      max_features=100000,
      ngram_range=(1, 1),
      stop_words=None,
      norm='l2', sublinear_tf=False,
      binary=True, use_idf=False)
    txt = tv.fit_transform(x)
    svd = TruncatedSVD(n_components=10, random_state=42, algorithm='arpack')
    svd = svd.fit(txt)
    svd_vectors = txt * csr_matrix(svd.components_).T
    svd_vectors = svd_vectors.toarray()
    svd_vectors = pd.DataFrame(svd_vectors)
    svd_vectors.columns = [var + str(i) for i, col in enumerate(svd_vectors.columns)]
    text[var] = svd_vectors
    for i in range(3): gc.collect()

[x.shape for x in text.values()]
text_full = pd.concat(list(text.values()), axis=1)
text_full.shape

for var in text_vars:
  dat = dat.drop(var, 1)

# Save
out = pd.concat([dat, text_full], axis=1)
out.to_csv('~/datasets/rental_train_svd_ind.csv', index=False)
