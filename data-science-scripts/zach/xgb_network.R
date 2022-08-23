##################################################
# Setup
##################################################
stop()
rm(list=ls(all=T))
gc(reset=T)
library(ggplot2)
library(data.table)
library(Metrics)
library(xgboost)

##################################################
# Load data
##################################################
data(diamonds)
dat <- data.table(diamonds)
rm(diamonds)

for(var in names(dat)){
  if(!is.numeric(var)){
    set(dat, j=var, value=as.numeric(dat[[var]]))
  }
}

set.seed(42)
idx = sample(c(T, F), nrow(dat), replace=TRUE, prob=c(.80, .20))
table(idx)/nrow(dat)

target = 'price'
normalized = log(dat[[target]])
normalized = (normalized - mean(normalized)) / sd(normalized)
set(dat, j=target, value=normalized)

dat_train <- dat[idx,]
dat_test <- dat[!idx,]

X_train <- dat[idx, setdiff(names(dat), target), with=F]
X_test <- dat[!idx, setdiff(names(dat), target), with=F]

y_train <- dat[idx,][[target]]
y_test <- dat[!idx,][[target]]

fit_xgb = function(X, y, nround=100, learning_rate=1, subsample=.80){
  d <- xgb.DMatrix(data=as.matrix(X), label=y)
  out <- list(
    rawxgb = xgboost(
    data = d,  
    nround = nround,
    learning_rate = learning_rate,
    subsample = subsample,
    base_score=0,
    verbose=T,
    objective = "reg:linear")
  )
  class(out) <- 'zachxgb'
  out
}

predict.zachxgb <- function(model, X){
  predict(model[['rawxgb']], xgb.DMatrix(data = as.matrix(X)))
}

##################################################
# Baseline models
##################################################

model_xgb <- fit_xgb(X_train, y_train, 10000, 0.001)
pred_xgb = predict(model_xgb, X_test)
print(rmse(pred_xgb, y_test))

model_lin <- lm(price ~ ., dat_train)
pred_lin <- predict(model_lin, dat_test)
print(rmse(pred_lin, y_test))

##################################################
# Orthogonal self learner
##################################################

# Initialize
N_models = 3
N_rows = nrow(X_train)
model_preds_train = matrix(rnorm(N_rows * N_models), ncol=N_models)
model_preds_train = svd(model_preds_train)$u

# Initial models
cor(model_preds_train)
cor(model_preds_train, y_train)
model_list = lapply(1:N_models, function(i){
  target = model_preds_train
  target[,i] = y_train
  target = rowMeans(target)
  fit_xgb(X_train, target, 10000, 0.001)
})
model_preds_train <- sapply(model_list, predict, X_train)
apply(model_preds_train, 2, function(x) rmse(x, y_train))
print(rmse(rowMeans(model_preds_train), y_train))

model_preds_test <- sapply(model_list, predict, X_test)
apply(model_preds_test, 2, function(x) rmse(x, y_test))
print(rmse(rowMeans(model_preds_test), y_test))
