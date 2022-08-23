import numpy as np
import pandas as pd

class MinMaxScaleCapper(object):
  def fit(self, y):
    self.min = y.min()
    self.range = y.max() - self.min
    return self

  def transform(self, y_new):
    out = (y_new - self.min) / self.range
    np.clip(out, 0, 1, out=out)
    return(out)


train = np.arange(-10, 0).astype('float64')

test = np.arange(-11, 2).astype('float64')
my_scalar = MinMaxScaleCapper()
my_scalar.fit(train)
print pd.DataFrame({
  'input': test,
  'output': my_scalar.transform(test).round(2)
})
