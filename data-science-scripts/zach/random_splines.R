# Setup
stop()
rm(list = ls(all=T))
gc(reset=T)

library(data.table)
library(scatterplot3d)
library(RcppEigen)
library(lspline)
library(glmnet)
rmse <- function(a, b) sqrt(mean((a-b)^2))

# Load data
dat_raw <- fread('~/workspace/data-science-scripts/zach/discontinuous.csv')

# Normalize
dat_raw[,a := scale(a, center = T, scale = T)]
dat_raw[,b := scale(b, center = T, scale = T)]

# Split train vs test
cut <- round(nrow(dat_raw)*.80)
dat_train <- dat_raw[1:cut,]
dat_test <- dat_raw[(cut+1):nrow(dat_raw),]
rm(dat_raw)

# Split X/y
X_train <- dat_train[,cbind(a, b)]
y_train <- dat_train[['target']]

X_test <- dat_test[,cbind(a, b)]
y_test <- dat_test[['target']]

# Plot
# dat_test[,scatterplot3d(a, b, target, pch = 16, grid=TRUE, box=FALSE, angle = 50)]

# Linear model - 0.30
# GBM (DataRobot) - 0.0049
model_linear <- glm(target ~ ., data=dat_train)
pred_linear <- predict(model_linear, dat_test)
print(rmse(pred_linear, y_test))

# Quantile spline basis
n_bins = 100
find_bins <- function(x, n) quantile(x, (1:n/n)[-n])
bins = apply(X_train, 2, find_bins, n_bins)

spline_basis <- function(x, knots){
  splines <- matrix(0, nrow=nrow(x), ncol=nrow(knots) * ncol(x) *2)
  i <- 0
  for(p in list(pmax, pmin)){
    for(var in 1:ncol(x)){
      for(k in 1:nrow(knots)){
        i <- i+1
        splines[,i] <- p(0, x[,var] - knots[k, var])
      }
    }
  }
  return(cbind(x, splines))
}
splines_train <- spline_basis(X_train, bins)
splines_test <- spline_basis(X_test, bins)

# Greedy quantile spline basis
pred <- mean(y_train)
train_internal <- splines_train

res <- apply(train_internal, 2, function(x){
  x = as.matrix(x)
  cf = RcppEigen::fastLmPure(x, y_train - pred, 3L)$coefficients
  c(rmse(x %*% cf + pred, y_train), cf)
})
best_idx <- which.min(res[1,])
best_cf <- res[2, best_idx]
best_rmse <- res[1, best_idx]
print(best_rmse)
print(res[,1:10])

pred <- pred + best_cf * splines_train[,best_idx]
train_internal <- apply(splines_train, 2, function(x) x * pred)

# Glmnet on spline basis - 0.07786828
model <- cv.glmnet(splines_train, y_train, alpha=1)
plot(model)
plot(model$glmnet.fit)
pred_spline <- predict(model, splines_test, s='lambda.min')
rmse(pred_spline, y_test)

# NLS
dt <- data.frame(dat_train)
dt[['target']] <- dt[['target']] + rnorm(nrow(dt), mean=0, sd=10)
nls_model <- nls(
  target ~ I(pmax(a - k1, 0)) + I(pmax(b - k2, 0)), 
  start=list(k1=1, k2=1), 
  data=dt,
  trace=T)
