library(quantmod)
library(forecast)
#TS_raw <- getSymbols('DGS10', src='FRED', auto.assign=FALSE)
#MN_raw <- getSymbols('DAAA', src='FRED', auto.assign=FALSE)
RATIO_raw <- getSymbols('AAA10Y', src='FRED', auto.assign=FALSE)

RATIO <- na.locf(RATIO_raw)
#RATIO <- RATIO["2008/"]
model <- ets(RATIO, model='ZZN')
plot(forecast(model, 5), include=30)
summary(model)

model2 <- auto.arima(RATIO, stepwise=FALSE, trace=TRUE)
plot(forecast(model2, 5), include=30)
summary(model2)
