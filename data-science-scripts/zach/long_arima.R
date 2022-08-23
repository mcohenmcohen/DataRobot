library(data.table)
library(forecast)

x = fread('https://s3.amazonaws.com/datarobot_public_datasets/time_series/hyndman_turkish_electricity_demand_train.csv')
turkish = x[['y']]
plot(turkish)

turkish = ts(turkish, frequency = 7)

model1 = auto.arima(turkish, trace=T, stepwise=F)


model2 = arima(turkish, c(100, 0, 0))
