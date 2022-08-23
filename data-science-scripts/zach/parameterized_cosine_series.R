
gc(reset=T)

library(data.table)
library(lbfgs)
library(numDeriv)
library(compiler)
library(glmnet)
library(forecast)
library(prophet)
library(RcppCNPy)
data("AirPassengers")

# train = data.frame(y=as.integer(AirPassengers))
# test = train

# train <- fread('~/datasets/upload/twitter_volume_over_time_train.csv')
# test <- fread('~/datasets/upload/twitter_volume_over_time_test.csv')

# train <- fread('~/datasets/iso_new_hourly_load_train.csv', colClasses=c("character", "character", "numeric"))
# test <- fread('~/datasets/iso_new_hourly_load_test.csv', colClasses=c("character", "character", "numeric"))

train <- fread('~/datasets/upload/hyndman_turkish_electricity_demand_train.csv')
test <- fread('~/datasets/upload/hyndman_turkish_electricity_demand_test.csv')

# train <- fread('~/datasets/upload/facebook_data_train.csv')
# test <- fread('~/datasets/upload/facebook_data_test.csv')

#https://stats.stackexchange.com/questions/60994/fit-a-sinusoidal-term-to-data
construct_basis <- cmpfun(function(t, x, include_t = F){
  n <- nrow(x)

  sin_ids <- 0*n + 1:n
  cos_ids <- 1*n + 1:n

  out <- matrix(0, nrow=length(t), ncol=length(x)*2)
  # t_mat <- t(matrix(t,nrow=n,ncol=length(t),byrow=TRUE))
  #
  # out[,sin_ids] <- sin(x*t_mat)
  # out[,cos_ids] <- cos(x*t_mat)

  for(j in 1:n){
    out[,j] <- sin(x[j,1]*t)
  }
  for(j in 1:n){
    out[,j+n] <- cos(x[j,2]*t)
  }

  if(include_t){
    return(cbind(t, out))
  }
  else(
    return(out)
  )

})

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

#######################################################
# Setup
#######################################################

y_raw <- train$y
y_test <- test$y
t <- seq_along(y_raw)
t_test <- max(t) + seq_along(y_test)

#######################################################
# NLS
#######################################################

#https://stats.stackexchange.com/questions/60994/fit-a-sinusoidal-term-to-data

#Remove a linear trend
#pred <- predict(lm(y_raw ~ t))

N <- 25
all_terms <- matrix(rep(0, N*2), ncol=2)
all_per <- rep(0, N)
pred <- rep(0, length(y_raw))
gc(reset=T)
i = 1
for(i in 1:N){
  y <- scale(y_raw - pred, center=T, scale=T)
  #y <- y_raw - pred
  ssp <- spec.pgram(y, plot=F)
  #i <- ssp$spec==max(ssp$spec)
  #points(ssp$freq[i], ssp$spec[i], col='red')
  per <- 1/ssp$freq[ssp$spec==max(ssp$spec)]
  all_per[i] <- per
  evalfun <- cmpfun(function(a){
    a[1]*sin(a[3]*pi/per*t)+a[2]*cos(a[4]*pi/per*t)
  })
  plotfun <- function(a, ...){
    plot(evalfun(a), ...)
  }
  mod <- nls(y ~ evalfun(c(a, b, F1, F2)), start=list(a=1, b=1, F1=2, F2=2), control=nls.control(warnOnly = T))
  all_terms[i,1] <- coef(mod)['F1']*pi/per
  all_terms[i,2] <- coef(mod)['F2']*pi/per
  pred1 <- evalfun(coef(mod))*attr(y, 'scaled:scale') + attr(y, 'scaled:center')
  #pred1 <- pred + evalfun(coef(mod))

  pred <- pred + pred1
  print(accuracy(pred, y_raw))

  plot(y_raw~t, type='l')
  lines(pred~t,col=5,lty=2, type='l')
  #lines(pred2~t,col=4,lty=2, type='l')
  title(main=i)
}
all_terms_r <- all_terms
all_terms
accuracy(pred, y_test)

#Load from python
if(F){
  mat <- scan('~/workspace/DataRobot/r.csv')
  mat <- matrix(mat, ncol = 2, byrow = TRUE)
  all_terms = mat
}

ssp <- spec.pgram(scale(y_raw), plot=F)
per <- 1/ssp$freq[order(ssp$spec, decreasing=T)]
round(all_per)
head(round(per), 7)

#glmnet
basis <- construct_basis(t, all_terms[rowSums(abs(all_terms)) != 0,,drop=F], include_t = T)
model_2 <- cv.glmnet(basis, y_raw, family=c('gaussian'), alpha=1)
pred2 <- predict(model_2, basis, s=c("lambda.min"))[,1]
plot(y_raw~t, type='l')
lines(pred2~t,col=5,lty=2, type='l')

#Predict for test set
basis_test <- construct_basis(t_test, all_terms[rowSums(abs(all_terms)) != 0,,drop=F], include_t = T)
pred2 <- predict(model_2, basis_test, s=c("lambda.1se"))[,1]
accuracy(pred2, y_test)
normalizedGini(pred2, y_test)
caret::R2(pred2, y_test)

#Smaller plot
#sub <- 24*7*4
sub <- 1440
#sub <- length(y_test)
plot(head(y_test, sub) ~ head(t_test, sub), type='l')
lines(head(pred2, sub) ~ head(t_test, sub),col=4,lty=2)
#lines(head(pred, sub) ~ head(t_test, sub),col=5,lty=2)

(all_terms - all_terms_r)

#######################################################
# Prophet
#######################################################

m <- prophet(data.table(
  ds = train$date,
  y = train$y
))

future <- make_future_dataframe(m, periods = length(test))
head(future)

forecast <- predict(m, data.table(ds = test$date))
head(forecast)

f_pred <- forecast$trend + forecast$weekly + forecast$weekly + forecast$yearly
head(f_pred)

accuracy(forecast$yhat, y_test)
plot(forecast$yhat, type='l')

plot(head(y_test, sub) ~ head(t_test, sub), type='l')
lines(head(forecast$yhat, sub) ~ head(t_test, sub),col=4,lty=2)

#######################################################
# BFGS
#######################################################

N <- 100
all_terms <- rep(0, N)
pred <- rep(0, length(y_raw))
for(i in 1:N){
  y <- scale(y_raw - pred, center=T, scale=T)
  ssp <- spec.pgram(y, plot=F)
  per <- 1/ssp$freq[ssp$spec==max(ssp$spec)]
  evalfun <- cmpfun(function(a){
    a[1]*sin(a[3]*pi/per*t)+a[2]*cos(a[3]*pi/per*t)
  })
  plotfun <- function(a, ...){
    plot(evalfun(a), ...)
  }
  optfun <- cmpfun(function(a){
    mean((y - evalfun(a)) ^ 2)
  })
  gradfun <- cmpfun(function(a){
    grad(optfun, a, method='simple')
  })

  init <- c(1,1,2)
  # plotfun(init)
  # optfun(init)
  # gradfun(init)

  best <- optim(init, optfun, method="BFGS")
  pred <- pred + evalfun(best$par)*attr(y, 'scaled:scale') + attr(y, 'scaled:center')
  plot(y_raw~t, type='l')
  lines(pred~t,col=4,lty=2)
  all_terms[i] <- best$par[3]*pi/per
}

#Fit glmnet
basis <- construct_basis(t, all_terms)
model_2 <- cv.glmnet(basis, y_raw, family=c('gaussian'))
plot(model_2$glmnet.fit)

#Predict for test set
basis_test <- construct_basis(t_test, all_terms)
pred <- predict(model_2, basis_test, s=c("lambda.min"))[,1]
accuracy(pred, y_test)

#######################################################
# One big nls
#######################################################

y <- scale(y_raw, center=T, scale=T)
ssp <- spec.pgram(y, plot=F)
per <- 1/ssp$freq[ssp$spec==max(ssp$spec)]

N <- 10

train_t_mat <- t(matrix(t,nrow=N,ncol=length(t),byrow=TRUE))

init <- c(rep(2, N*2), rep(1, N*2))
names(init) <- paste0('V', 1:(N*4))

sin_ids <- 0*N + 1:N
cos_ids <- 1*N + 1:N
per_ids <- c(2*N + 1:N, 3*N + 1:N)
update <- function(t_mat, ...){
  x <- c(...)
  basis <- matrix(0, nrow=nrow(t_mat), ncol=N*2)
  basis[,sin_ids] <- sin(x[sin_ids]*pi/per*t_mat)
  basis[,cos_ids] <- sin(x[cos_ids]*pi/per*t_mat)
  tcrossprod(basis, t(x[per_ids]))[,1]
}
update(train_t_mat, init)
xvars <- paste(names(init), collapse=', ')
flma <- as.formula(paste('y ~ update(train_t_mat,', xvars, ')'))
mod <- nls(flma, start=init, control=nls.control(warnOnly = T))

#######################################################
# Scratch pad
#######################################################

plot(y)

reslm2 <- lm(y ~ sin(2*pi/per*t)+cos(2*pi/per*t)+sin(4*pi/per*t)+cos(4*pi/per*t))
summary(reslm2)
lines(fitted(reslm2)~t,col=3)

evalfun <- cmpfun(function(a){
  1 / a[2] * cos(x*a[1])
})
plotfun <- function(a, ...){
  plot(evalfun(a), ...)
}
optfun <- cmpfun(function(a){
  mean((y - eval_fun(a)) ^ 2)
})
gradfun <- cmpfun(function(a){
  grad(optfun, a, method='simple')
})

init <- c(1/length(y), diff(range(y)))
plotfun(init)
optfun(init)
gradfun(init)

best <- optim(init, optfun, method="BFGS")
plotfun(best$par)
plotfun(best$par, type='l', col='red')
lines(y/25)
plot(y)
plot(y - evalfun(best$par)*10000)
optfun(best$par)
gradfun(best$par)

best <- lbfgs(optfun, gradfun, init)
