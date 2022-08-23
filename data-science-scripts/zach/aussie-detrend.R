
########################################
# Load data and train/test split
########################################
library(fpp)
library(forecast)
plot(austourists)
train <- window(austourists, end=c(2008,4))
test <- window(austourists, start=2009)

########################################
# Ets model
########################################
fit <- auto.arima(train, stepwise=FALSE)
plot(forecast(fit, h=8))
lines(test, col='red')

p <- forecast(fit, h=8)$mean
a <- window(austourists, start=2009)
pct_error_arima <- abs(p - test) / test
mean(pct_error_arima)
round(pct_error_arima, 2)

########################################
# Assembe datarobot dataset
########################################

#Detrend
linear_trend_model <- tslm(train ~ season + trend)
trend_coef <- coef(linear_trend_model)[['trend']]
linear_trend <- 1:length(austourists) * trend_coef
austourists_detrend <- austourists - linear_trend
plot(austourists_detrend)

#Seasonal dummies + linear time
season <- seasonaldummy(austourists)
time <- 1:length(austourists)

#DataRobot time-based validaiton set
set=c(
  rep("train", 8*4),
  rep("valid", 2*4),
  rep("holdout", 2*4)
)

#Assemble data.frame and save train/test
dat <- data.frame(
  y = as.numeric(austourists_detrend),
  season,
  time,
  set
)
write.csv(dat[dat$set!='holdout',], '~/datasets/austourists_train.csv', row.names=F)
write.csv(dat[dat$set=='holdout',], '~/datasets/austourists_test.csv', row.names=F)

########################################
# Run datarobot here and download predictions
########################################

#Use advanced options > partition column, use "set" for the partitioning
#Make sure you re-run the best model on 100%

########################################
# Look at DataRobot results
########################################
p <- read.csv("~/Downloads/aussie_new_Support_Vector_Regressor_(Radial_Kernel)_(20)_100_Informative_Features_austourists_test.csv")$Prediction
p <- p + linear_trend[41:48]
pct_error_dr <- abs(p - test) / test
mean(pct_error_dr)
round(pct_error_dr, 2)

plot(austourists)
pred_full <- ts(c(rep(NA, 40), p), start=1999, frequency=4)
lines(pred_full, type='l', col='blue')
lines(test, col='red')

########################################
# Compare arima to dr
########################################

plot(pct_error_arima)
lines(pct_error_dr, col='red')
legend(2010.5, 0.08, c("arima", "DR"), lty=c(1,1), lwd=c(2.5,2.5),col=c("black","red"))
