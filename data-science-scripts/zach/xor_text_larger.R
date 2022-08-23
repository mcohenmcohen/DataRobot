stop()
rm(list=ls(all=T))
gc(reset=T)

library(data.table)
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

dat[,table(target, num1)/.N]
dat[,table(target, num2)/.N]
dat[,table(target, num1, num2)/.N]

dat[which(num2), diag_1_desc := paste(diag_1_desc, 'MAGICWORD')]
dat[,num2 := NULL]

fwrite(dat, '~/datasets/xor_text.csv')