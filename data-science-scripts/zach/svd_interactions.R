stop()
rm(list=ls(all=T))
gc(reset=T)

library(data.table)
library(Matrix)
library(irlba)
library(glmnet)
library(caTools)
library(Metrics)

# Setup sparse Xor
rows = 1e4
cols = 1e2
num1 = sample(c(T, F), rows, replace=T)
num2 = sample(c(T, F), rows, replace=T)
target = xor(num1, num2)
mat <- rsparsematrix (nrow=rows, ncol=cols, nnz=.10*rows*cols)
mat <- cbind(mat, cbind(as.integer(num1), as.integer(num2)))

# Split train/test
split = .80 * rows
mat_train <- mat[1:split,]
mat_test <- mat[(split+1):rows,]
stopifnot(nrow(mat_train) + nrow(mat_test) == nrow(mat))

target_train <- target[1:split]
target_test <- target[(split+1):rows]
stopifnot(length(target_train) + length(target_test) == length(target))

# SVD via irlba
model = irlba(mat, nv=0, nu=16)

# Interactions
mat_inter = matrix(0, nrow=nrow(model$u), ncol=ncol(model$u)^2)
col=1
for(i in 1:ncol(model$u)){
  for(j in i:ncol(model$u)){
    mat_inter[,col] <- model$u[,i] * model$u[,j]
    col = col + 1
  }
}

# Filter
mat_inter <- mat_inter[,colSums(abs(sign(mat_inter))) > 0]

# Split train/test
mat_inter_train <- mat_inter[1:split,]
mat_inter_test <- mat_inter[(split+1):rows,]
stopifnot(nrow(mat_inter_train) + nrow(mat_inter_test) == nrow(mat_inter))

# Model
library(doParallel)
registerDoParallel(4)
model <- cv.glmnet(cbind(mat_train, mat_inter_train), target_train, family="binomial", alpha=1, nfolds=5, parallel=T)

# Predict
s <- "lambda.min"
p_train <- predict(model, cbind(mat_train, mat_inter_train), s = s, type='response')
p_test  <- predict(model, cbind(mat_test,   mat_inter_test), s = s, type='response')

colAUC(p_train, target_train)
colAUC(p_test, target_test)

logLoss(target_train, p_train)
logLoss(target_test, p_test)
