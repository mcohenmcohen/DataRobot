import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

url = "https://s3.amazonaws.com/datarobot_public_datasets/time_series/AirPassengers.csv"
dat = pd.read_csv(url)
dat['date'] = pd.to_datetime(dat['date'])
dat = dat.set_index('date')
train = dat['1949-01-01':'1958-12-01']
test = dat['1959-01-01':'1960-12-01']

mod = SARIMAX(
  train,
  order=(4, 1, 4), seasonal_order=(1, 0, 1, 12),
  enforce_stationarity=False,
  enforce_invertibility=False)
res = mod.fit()
test.loc[:,'pred'] = res.forecast(24)
test[['y', 'pred']].plot(figsize=(12, 8))

plt.show()
