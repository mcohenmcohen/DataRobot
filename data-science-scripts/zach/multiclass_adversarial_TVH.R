library(data.table)
library(readr)
dat = fread('https://s3.amazonaws.com/datarobot_public_datasets/mnist_train.csv')

set.seed(42)
dat[,user_partition := sample(c('T', 'V', 'H'), .N, replace=T, prob=c(.80, .10, .10))] # 80% train, 10% test 10% valid
dat[,table(user_partition) / .N]

dat[target == 1, user_partition := 'V'] # Move a class entirely to validation
dat[target == 2, user_partition := 'H'] # Move a class entirely to holdout

#Look at class distribution
dat[,table(user_partition) / .N]
dat[,table(target, user_partition) / .N]

# Save
write_csv(dat, '~/datasets/mnist_train_new_class.csv')
