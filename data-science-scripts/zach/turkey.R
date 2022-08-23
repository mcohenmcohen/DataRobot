# (c) Turkish electricity demand data.
# Daily data from 1 January 2000 to 31 December 2008.
# telec <- read.csv("https://robjhyndman.com/data/turkey_elec.csv")
# telec <- msts(telec, start=2000, seasonal.periods = c(7,354.37,365.25))
# model <- tbats(telec, num.cores=8)
# plot(forecast(model, 365), include=365)

stop()
rm(list=ls(all=T))
gc(reset=T)
normalizedGini <- function(aa, pp) {
  Gini <- function(a, p) {
    if (length(a) !=  length(p)) stop("Actual and Predicted need to be equal lengths!")
    temp.df <- data.frame(actual = a, pred = p, range=c(1:length(a)))
    temp.df <- temp.df[order(-temp.df$pred, temp.df$range),]
    population.delta <- 1 / length(a)
    total.losses <- sum(a)
    null.losses <- rep(population.delta, length(a)) # Hopefully is similar to accumulatedPopulationPercentageSum
    accum.losses <- temp.df$actual / total.losses # Hopefully is similar to accumulatedLossPercentageSum
    gini.sum <- cumsum(accum.losses - null.losses) # Not sure if this is having the same effect or not
    sum(gini.sum) / length(a)
  }
  Gini(aa,pp) / Gini(aa,aa)
}
library(data.table)
library(forecast)
train <- fread('~/datasets/time_series/train/hyndman_turkish_electricity_demand_train.csv')
test <- fread('~/datasets/time_series/test/hyndman_turkish_electricity_demand_test.csv')
model <- tbats(train$y, num.cores=7, seasonal.periods=c(7, 354.37, 365.25))
fcast <- forecast(model, nrow(test))
accuracy(fcast$mean, test$y)
normalizedGini(test$y, fcast$mean)

#                ME     RMSE      MAE       MPE     MAPE
# Test set -581.4826 1578.197 941.7062 -3.413157 4.903234
# 0.8035122 Gini
