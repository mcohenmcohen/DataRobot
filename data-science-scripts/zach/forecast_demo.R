#Setup
#install.packages('forecast')
library(forecast)
data("AirPassengers")
plot(AirPassengers)
str(AirPassengers)
print(AirPassengers)

as.integer(AirPassengers)
ts(AirPassengers, frequency = 1)

train <- window(AirPassengers, end=c(1959, 12))
test <- window(AirPassengers, start=c(1960, 1))

#Arima models
arima_model <- auto.arima(train, stepwise=FALSE, trace=TRUE)
f_arima <- forecast(arima_model, 12)
plot(f_arima, include=24)
lines(test, col='red')
accuracy(f_arima, test)

#Exponential smoothing models
ets_model <- ets(train, restrict=FALSE)
f_ets <- forecast(ets_model, 12)
plot(f_ets, include=24)
lines(test, col='red')
accuracy(f_ets, test)

#Multiple seasonal periods
# (c) Turkish electricity demand data.
# Daily data from 1 January 2000 to 31 December 2008.
telec <- read.csv("http://robjhyndman.com/data/turkey_elec.csv")
telec <- msts(telec, start=2000, seasonal.periods = c(7,354.37,365.25))

train <- window(telec, end=c(2007, 365))
test <- window(telec, start=c(2008, 1))

arima_model <- auto.arima(train, stepwise=TRUE, trace=TRUE)
f_arima <- forecast(arima_model, length(test))
plot(f_arima, include=90)
lines(test, col='red')
accuracy(f_ets, test)

ets_model <- ets(train, restrict=FALSE)
f_ets <- forecast(ets_model, length(test))
plot(f_ets, include=90)
lines(test, col='red')
accuracy(f_ets, test)

tbats_model <- tbats(train)
f_tbats <- forecast(tbats_model, length(test))
plot(f_tbats, include=90)
lines(test, col='red')
accuracy(f_tbats, test)

library(hts)
bts <- ts(5 + matrix(sort(rnorm(500)), ncol=5, nrow=100))
y <- hts(bts, nodes=list(2, c(3, 2)))
