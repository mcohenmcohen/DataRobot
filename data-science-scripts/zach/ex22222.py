from sklearn.base import BaseEstimator, TransformerMixin


def ConcatenateFeatures(X):
  # TODO: FILL THIS IN
  pass

class CustomTransformer(BaseEstimator, TransformerMixin):
  
  def fit(self, X, y = None):
    only2 = pd.DataFrame(X.nunique()).reset_index()
    only2 = only2.rename(columns={"index":"features",0:"levels"} )
    self.featuresWith2levels = only2[only2["levels"]==2].features
    return self

    def transform(self, X, y = None):
      combinedFeatures = ConcatenateFeatures(X[self.featuresWith2levels])
      return combinedFeatures
