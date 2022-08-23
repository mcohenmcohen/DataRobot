from __future__ import division
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix

from tesla.utils import sparse_polynomial_features as spf
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import rand
np.random.seed(42)

def tuple_to_csr(csr_tuple):
    """Wraps conversion of cython produced tuple into scipy csr.
    Args:
        csr_tuple: data, column index, and row index, matrix shape
    Returns:
        scipy csr object.
    """
    csr = csr_matrix((csr_tuple[0], csr_tuple[1], csr_tuple[2]), shape=csr_tuple[3])
    return csr

class SparsePolynomialFeaturizer(BaseEstimator, TransformerMixin):
  
    def __init__(self, degree=2):
        self.degree = degree

    def fit(self, X):
      return self

    def transform(self, X):
        if self.degree == 2:
            poly_features = tuple_to_csr(spf.sec_deg_poly_feats(X))
        elif self.degree == 3:
            poly_features = tuple_to_csr(spf.third_deg_poly_feats(X))
        return poly_features

class FeatureFilter(BaseEstimator, TransformerMixin):
  
    def __init__(self, pct=0):
        self.pct = pct
        self.mask = None

    def fit(self, X):
      self.mask = np.array(((X != 0).mean(axis=0) > self.pct)).ravel()
      return self

    def transform(self, X):
      return X[:, self.mask]
    
    
# Make a csr
dat = rand(10, 3, density=0.2, format='csr')
dat = np.round(dat, 1)

# Find poly features
PolyFeatures = SparsePolynomialFeaturizer(degree=2)
PolyFeatures.fit(dat)
dat_poly = PolyFeatures.transform(dat)

# Filter features
FilteredFeatures = FeatureFilter(pct=.10)
FilteredFeatures.fit(dat_poly)
dat_poly_filter = FilteredFeatures.transform(dat_poly)

# Print data
print(np.round(dat.todense(), 2))
print(np.round(dat_poly.todense(), 2))
print(np.round(dat_poly_filter.todense(), 2))

# Print shapes
print(dat.shape)
print(dat_poly.shape)
print(dat_poly_filter.shape)
