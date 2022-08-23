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
train <- fread('~/datasets/time_series/train/iso_new_hourly_load_train.csv')
test <- fread('~/datasets/time_series/test/iso_new_hourly_load_test.csv')
model <- tbats(train$y, num.cores=7, seasonal.periods=c(1, 7, 365.25)*24)
fcast <- forecast(model, nrow(test))
accuracy(fcast$mean, test$y)
normalizedGini(test$y, fcast$mean)

#               ME     RMSE      MAE    MPE     MAPE
# Test set 396.1384 2240.049 1849.176 3.0655 13.49595
#Gini: 0.7616298
