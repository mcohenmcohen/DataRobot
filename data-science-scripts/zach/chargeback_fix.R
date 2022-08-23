rm(list=ls(all=T))
gc(reset(all=T))
set.seed(42)
library(data.table)
x_raw = fread('https://s3.amazonaws.com/datarobot_public_datasets/chargeback_clean_80.csv')
y_raw = fread('https://s3.amazonaws.com/datarobot_public_datasets/chargeback_clean_20.csv')
x = copy(x_raw)
y = copy(y_raw)
x[is.na(as.numeric(postalCode)),table(postalCode, chargeback)]
y[is.na(as.numeric(postalCode)),table(postalCode, chargeback)]
x[,postalCode := paste0('z', postalCode)]
y[,postalCode := paste0('z', postalCode)]
fwrite(x, '~/datasets/chargebackl_clean_fixed_80.csv')
fwrite(y, '~/datasets/chargeback_clean_fixed_20.csv')
