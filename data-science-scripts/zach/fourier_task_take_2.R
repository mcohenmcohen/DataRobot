rm(list=ls(all=T))
gc(reset=T)

library(data.table)
library(lbfgs)
library(numDeriv)
library(compiler)
library(glmnet)
library(forecast)
library(prophet)
library(RcppCNPy)
library(xgboost)
library(timeDate)
library(splines)
data("AirPassengers")

# Todo: solve seasonality (up to 5) using bfgs + spline
# https://www.rdatagen.net/post/generating-non-linear-data-using-b-splines/

# train <- fread('~/datasets/iso_new_hourly_load_train.csv', colClasses=c("character", "character", "numeric"))
# test <- fread('~/datasets/iso_new_hourly_load_test.csv', colClasses=c("character", "character", "numeric"))

train <- fread('~/datasets/upload/hyndman_turkish_electricity_demand_train.csv')
test <- fread('~/datasets/upload/hyndman_turkish_electricity_demand_test.csv')

# train <- fread('~/datasets/upload/facebook_data_train.csv')
# test <- fread('~/datasets/upload/facebook_data_test.csv')

# train <- fread('https://s3.amazonaws.com/datarobot_public_datasets/time_series/facebook_retail_train.csv')
# train <- fread('https://s3.amazonaws.com/datarobot_public_datasets/time_series/facebook_retail_train.csv')

# train <- fread('https://s3.amazonaws.com/datarobot_public_datasets/time_series/hyndman_usa_gasoline_train.csv')
# test <- fread('https://s3.amazonaws.com/datarobot_public_datasets/time_series/hyndman_usa_gasoline_test.csv')

#######################################################
# Setup
#######################################################

train[,date := as.Date(date)]
test[,date := as.Date(date)]
train[,plot(y ~ date, type='l')]

start <- train[,min(date)]

train_y <- train[['y']]
test_y <- test[['y']]

#######################################################
# Crazy bfgs model
#######################################################

#https://www.rdatagen.net/post/generating-non-linear-data-using-b-splines/
spline_basis <- cmpfun(function(t, knots){
  bs(
    x = t, knots = knots,
    degree = 1,
    Boundary.knots = range(t),
    intercept = TRUE
    )
})

#https://stats.stackexchange.com/questions/60994/fit-a-sinusoidal-term-to-data
fourier_basis <- cmpfun(function(t, periods, harmonics, start){
  fourier <- matrix(rep(0, length(t)*length(periods)*harmonics*2), nrow=length(t))
  col_index <- 0
  names <- paste0('V', 1:ncol(fourier))
  for(p in periods){
    for(h in 1:harmonics){
      for(f in c('sin', 'cos')){
        col_index <- col_index + 1
        fourier[,col_index] <- get(f)(h*2*pi/p*t)
        names[col_index] <- paste0(f,'_h', h, '_p', p)
      }
    }
  }
  fourier <- data.table(fourier)
  setnames(fourier, names)
  return(as.matrix(fourier))
})

seasonal_spline <- function(x, y, start){

  t <- as.integer(x-start)

  basis <- cmpfun(function(pars){
    knots <- pars[1:3]
    periods <- pars[4:6]
    theta <- pars[7:length(pars)]

    basis <- cbind(
      spline_basis(t, knots),
      fourier_basis(t, periods, harmonics=2, start)
    )
  })

  fitmodel <- cmpfun(function(pars){
    pred <- basis(par) %*% theta
    sqrt(mean((y - pred)^2))
  })

  init <- c(
    quantile(t, c(.25, .50, .75)),
    c(365.25, 30, 7),
    rep(0, 17)
  )

  min <- c(
    quantile(t, c(.1, .2, .3)),
    c(6, 4, 2),
    rep(-Inf, 17)
  )

  max <- c(
    quantile(t, c(.7, .8, .9)),
    c(5000, 4000, 3000),
    rep(Inf, 17)
  )

  res <- optim(
    init, fitmodel, method='BFGS',
    lower=min,
    upper=max)

  res

  new_basis <- basis(res$par)

}

#######################################################
# Add linear spline trend
#######################################################
y <- train_y
t <- as.integer(train[['date']]-start)
library(strucchange)
plot(y~t)
mod <- strucchange::breakpoints(y~t)
model_base <- lm(y ~ ., data.table(y, spline_basis(t, mod$breakpoints)))
lines(predict(mod2), col='red')

train_y <- train_y - predict(model_base, train)

#######################################################
# Search for an extra basis using a linear model
#######################################################

my_basis <- c(365.25)
my_harmonics <- 2
for(i in 1:25){
  train_x <- construct_basis(train, my_basis, my_harmonics, start)
  model <- cv.glmnet(train_x, train_y, type.measure='mse', nfolds=10)
  resid <- train_y - predict(model, train_x)[,1]
  resid <- scale(resid, center=T, scale=T)
  plot(resid, type='l')
  ssp <- spec.pgram(resid, plot=F)
  per <- 1/ssp$freq[which.max(ssp$spec)]
  my_basis <- c(my_basis, per)
}
my_basis

my_basis <- unique(my_basis)
train_x <- construct_basis(train, my_basis, my_harmonics, start)
test_x  <- construct_basis(test , my_basis, my_harmonics, start)

#######################################################
# Model & predict
#######################################################

t <- as.integer(test[['date']]-start)
pred_base <- predict(model_base, data.table(spline_basis(t, mod$breakpoints)))

model <- cv.glmnet(train_x, train_y)

pred <- pred_base + predict(model, test_x)
pred <- pred[,1]

#######################################################
# Model & predict
#######################################################

train[,yr := year(date)]
year <- train[,sort(unique(yr))]
folds <- pbsapply(year, function(x){
  train[,which(yr %in% x)]
})
train[,yr := NULL]
model <- xgb.cv(data = train_x, label = train_y, max_depth = 8, eta = 0.05, nthread = 8, nrounds = 1000, objective = "reg:linear", folds = folds)
best <- which.min(model$evaluation_log$test_rmse_mean)
model <- xgboost(data = train_x, label = train_y, max_depth = 8, eta = 0.05, nthread = 8, nrounds = best, objective = "reg:linear")

pred_base <- predict(model_base, test)
pred <- pred_base + predict(model, test_x)

#######################################################
# Accuracy
#######################################################

normalizedGini <- function(pp, aa) {
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
accuracy(pred, test[['y']]) #RMSE = 1507.526, MAE=1078.643
normalizedGini(pred, test[['y']]) #0.8012909
caret::R2(pred, test[['y']]) #0.5600059

# Small
# sub <- 1440
# plot(head(test[['y']], sub) ~ I(1:sub), type='l')
# lines(head(pred, sub) ~ I(1:sub), col=4, lty=2)

# Full
plot(test[['y']] ~ I(1:nrow(test)), type='l')
lines(pred ~ I(1:nrow(test)), type='l', col='blue', lty=2)
