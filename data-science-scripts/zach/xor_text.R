stop()
rm(list=ls(all=T))
gc(reset=T)

library(data.table)
library(Matrix)
library(caTools)
library(irlba)
library(text2vec)

dat_raw <- fread('https://s3.amazonaws.com/datarobot_public_datasets/sas/10k_diabetes.csv ')

set.seed(42)
dat <- dat_raw[,list(
  diag_1_desc, 
  num1 = sample(c(T, F), .N, replace=T),
  num2 = sample(c(T, F), .N, replace=T)
)]

dat[,table(num1)/.N]
dat[,table(num2)/.N]
dat[,table(num1, num2)/.N]

dat[,target := xor(num1, num2)]
#dat[sample(1:.N, .N/10), target := !target]

dat[,table(target, num1)/.N]
dat[,table(target, num2)/.N]
dat[,table(target, num1, num2)/.N]

dat[which(num2), diag_1_desc := paste(diag_1_desc, 'MAGICWORD')]

# See if SVD solves xor
mat <- dat[,Matrix(cbind(
  as.numeric(num1), 
  as.numeric(num2),
  sample(0:1, .N, replace=T),
  sample(0:1, .N, prob=c(.8, .2), replace=T),
  sample(0:1, .N, prob=c(.2, .8), replace=T)
))]
mat_train <- mat[1:8000,]
mat_test <- mat[8001:10000,]

interaction_count <- crossprod(mat_train)

targ_diag <- dat[1:8000,Diagonal(x=as.numeric(target))]
train_X_target <- t(crossprod(mat_train, targ_diag))
interaction_sum <- crossprod(train_X_target)

interaction <- interaction_sum/interaction_count
interaction <- dat[,mean(target)] - interaction

interaction_count
interaction_sum
interaction

#interaction <- scale(interaction, center=T, scale=T)

n_factor <- 5
model <- svd(interaction, nu=0, nv=n_factor)
pred_matrix <- model$v %*% Diagonal(x=model$d[1:n_factor])

pred_train_partial <- mat_train %*% pred_matrix
pred_test_partial <- mat_test %*% pred_matrix

pred_train <- rep(0, nrow(pred_train_partial))
pred_test <- rep(0, nrow(pred_test_partial))
for(i in 1:n_factor){
  for(j in 1:i){
    pred_train <- pred_train + pred_train_partial[,i]*pred_train_partial[,j]
    pred_test <- pred_test + pred_test_partial[,i]*pred_test_partial[,j]
  }
}
colAUC(pred_train, dat[1:8000,target])
colAUC(pred_test, dat[8001:10000,target])

colAUC(pred_train_partial, dat[1:8000,target])
colAUC(pred_test_partial, dat[8001:10000,target])

colAUC(pred_train_partial*pred_train_partial, dat[1:8000,target])
colAUC(pred_test_partial*pred_test_partial, dat[8001:10000,target])

# Save
dat[,num2 := NULL]
fwrite(dat, '~/datasets/xor_text.csv')