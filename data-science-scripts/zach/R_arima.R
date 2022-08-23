
#AirPassengers
library(forecast)
data("AirPassengers")
train = window(AirPassengers, start=c(1949, 1), end=c(1958, 12))
test = window(AirPassengers, start=c(1959, 1), end=c(1960, 12))
forecast_wrapper <- function(model, ...){
  mod <- model(train, ...)
  pred <- forecast::forecast(mod, length(test))
  plot(pred, ylim = range(c(train, test)), PI=FALSE, shaded=FALSE)
  lines(train, col='black')
  lines(test, col='black')
  lines(fitted(mod), col='blue', lty=2)
  forecast::accuracy(pred, test)[,'MAE']
}
forecast_wrapper(forecast::meanf)
forecast_wrapper(forecast::bats, seasonal.periods=12)
forecast_wrapper(forecast::auto.arima, stepwise=FALSE)
forecast_wrapper(forecast::ets, restrict=FALSE, allow.multiplicative.trend=TRUE)
forecast_wrapper(forecast::tbats, seasonal.periods=12)
tr <- data.frame(
  date = seq_along(train),
  AirPassengers = train
)
te <- data.frame(
  date = length(train) + seq_along(test),
  AirPassengers = test
)
stopifnot((nrow(tr) + nrow(te)) == length(AirPassengers))
write.csv(tr, '~/datasets/AirPassengers_train.csv', row.names=FALSE)
write.csv(te, '~/datasets/AirPassengers_test.csv', row.names=FALSE)

#Nutonian
#https://trial.nutonian.com/DataRobot/modeling/AirPassengers_train_79d9b7e7-1844-4f0a-9e42-56a7176d881b/0
mylag <- function(x, n){
  tmp <- quantmod::Lag(as.numeric(x), n)[,1]
  tmp <- ts(tmp, start=start(x), frequency = frequency(x))
  return(tmp)
}
pred1 <- function(x){
  36.98 + 1.395*pmax(mylag(x, 24), mylag(x, 33)) - 0.3684*mylag(x, 33)
}
pred2 <- function(x){
  43.85 + 1.083*mylag(x, 24)
}
eq_pred <- pred2(AirPassengers)
eq_pred <- ts(eq_pred, start=start(AirPassengers), frequency = frequency(AirPassengers))
eq_pred_train <- window(eq_pred, start=start(train), end=end(train))
eq_pred_test <- window(eq_pred, start=start(test), end=end(test))
plot(AirPassengers, col='white')
lines(train, col='black')
lines(test, col='black')
lines(eq_pred_train, col='blue', lty=2)
lines(eq_pred_test, col='blue', lty=1)
accuracy(eq_pred_test, test)

#Lynx
library(forecast)
data("lynx")
train <- window(lynx, end=1900)
test <- window(lynx, start=1901)
write.csv(train, '~/datasets/lynx_train.csv')
write.csv(test, '~/datasets/lynx_test.csv')
forecast_wrapper(forecast::ets, restrict=FALSE, allow.multiplicative.trend=TRUE)
forecast_wrapper(forecast::meanf)
forecast_wrapper(forecast::bats, seasonal.periods=9.63)
forecast_wrapper(forecast::auto.arima, stepwise=FALSE)
forecast_wrapper(forecast::auto.arima, stepwise=FALSE, max.p=0, max.q=5)
forecast_wrapper(forecast::tbats, seasonal.periods=9.63)
