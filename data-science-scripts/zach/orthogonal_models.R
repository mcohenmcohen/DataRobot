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

dtrain <- xgb.DMatrix(data = as.matrix(X_train), label=y_train)
dtest <- xgb.DMatrix(data = as.matrix(X_test), label=y_test)

##################################################
# Baseline models
##################################################

model_xgb <- xgboost(
  data = dtrain,  
  nround = 100,
  learning_rate = 1,
  subsample = .80,
  base_score=0,
  objective = "reg:linear")

pred_xgb = predict(model_xgb, dtest)
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
random_preds = matrix(rnorm(N_rows * N_models), ncol=N_models)
random_orthogonal = svd(random_preds)$u 
model_inits = y_train * random_orthogonal

# Initial models
model_preds_train = model_inits
model_preds_test = matrix(0, nrow=nrow(X_test), ncol=N_models)
model_list = lapply(1:N_models, function(x) NULL)
for(i in 1:N_models){
  dat_loop <- xgb.DMatrix(as.matrix(X_train), label=model_inits[,i])
  model_list[[i]] <- xgboost(
    data = dat_loop,  
    nround = 100,
    learning_rate = 1,
    subsample = .80,
    base_score=0,
    objective = "reg:linear")
  model_preds_train[,i] <- predict(model_list[[i]], xgb.DMatrix(as.matrix(X_train)))  # Todo CV or use RF
  model_preds_test[,i] <- predict(model_list[[i]], xgb.DMatrix(as.matrix(X_test)))  # Todo CV or use RF
}

# Final model
dat_loop <- xgb.DMatrix(cbind(as.matrix(X_train), model_preds_train), label=y_train)
final_model <- xgboost(
  data = dat_loop,  
  nround = 100,
  learning_rate = 1,
  subsample = .80,
  base_score=0,
  objective = "reg:linear")

pred_test = predict(final_model, xgb.DMatrix(cbind(as.matrix(X_test), model_preds_test)))
print(rmse(pred_test, y_test))

