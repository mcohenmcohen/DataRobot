library(data.table)
set.seed(42)
x = fread('~/datasets/10kDiabetes.csv')
x[,partition := sample(c('T', 'V', 'H'), .N, prob=c(.8, .1, .1), replace=T)]
x[readmitted == TRUE, partition := sample(c('V', 'H'), .N, replace=T)]
x[readmitted == FALSE & partition == 'T', readmitted := c(rep(TRUE, 1), rep(FALSE, .N-1))]
x[,table(partition, readmitted, useNA='ifany')]
write.csv(x, '~/datasets/10kDiabetes_bad_partition.csv')
