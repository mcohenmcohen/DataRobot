#Setup
library('data.table')
library('readr')
data("lynx")
train <- window(lynx, end=1900)
test <- window(lynx, start=1901)

#Prophet
library(prophet)
prophet_wrapper <- function(...){
  mod <- prophet::prophet(data.frame(ds=1:80,y=as.numeric(train)), ...)
  future <- prophet::make_future_dataframe(mod, periods=length(test))
  pred <- ts(predict(mod, future)[1:34,"yhat"], start=start(test), frequency = 1)
  plot(train, ylim = range(lynx), xlim=c(start(lynx)[1], end(lynx)[1]))
  lines(pred, col='blue')
  plot(pred, ylim = range(lynx), xlim=c(1,length(lynx)), type='l')
  lines(test, col='black')
  forecast::accuracy(pred, test)[,'RMSE']
}
prophet_wrapper()

#Forecast
#Forecast and prophet don't play nice
library(forecast)

forecast_wrapper <- function(model, ...){
  mod <- model(train, ...)
  pred <- forecast::forecast(mod, length(test))
  plot(pred, ylim = range(lynx), PI=FALSE, shaded=FALSE)
  lines(train, col='black')
  lines(test, col='black')
  lines(fitted(mod), col='blue', lty=2)
  forecast::accuracy(pred, test)[,'RMSE']
}

forecast_wrapper(forecast::ets, restrict=FALSE, allow.multiplicative.trend=TRUE)
forecast_wrapper(forecast::meanf)
forecast_wrapper(forecast::bats, seasonal.periods=9.63)
forecast_wrapper(forecast::auto.arima, stepwise=FALSE)
forecast_wrapper(forecast::auto.arima, stepwise=FALSE, max.p=0, max.q=5)
forecast_wrapper(forecast::tbats, seasonal.periods=9.63)

#Datarobot
K <- 4
f_train <- fourier(ts(train, frequency=9.63), K)
f_test <- fourier(ts(train, frequency=9.63), K, length(test))
train_dr <- data.table(
  lynx = train,
  date = as.Date(paste(start(train)[1]:end(train)[1] + 80, 1, 1, sep='-')),
  f_train
)
test_dr <- data.table(
  lynx = test,
  date = as.Date(paste(start(test)[1]:end(test)[1] + 80, 1, 1, sep='-')),
  f_test
)
train_dr <- rbind(train_dr, train_dr)
write_csv(train_dr, '~/datasets/lynx_train.csv')
write_csv(test_dr, '~/datasets/lynx_test.csv')

#Upload to DR here and download preds
dr_pred <- fread('~/Downloads/Untitled_Project_TensorFlow_Multilayer_Perceptron_Regressor_(29)_0_f1_lynx_test (1).csv')
pred <- ts(dr_pred$Prediction, start=start(test), frequency = 1)
plot(train, ylim = range(lynx), xlim=c(start(lynx)[1], end(lynx)[1]))
lines(pred, col='blue')
lines(test, col='black')
forecast::accuracy(pred, test)[,'RMSE']
