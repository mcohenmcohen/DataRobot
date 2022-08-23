library(data.table)
dat <- fread('~/datasets/10kDiabetes.csv')
dat[,diag_2_desc := NULL]
dat[,diag_3_desc := NULL]
fwrite(dat, '~/datasets/10kDiabetes_one_text.csv')

dat <- fread('~/datasets/10kDiabetes.csv')
dat[,diag_3_desc := NULL]
fwrite(dat, '~/datasets/10kDiabetes_two_text.csv')

dat <- fread('~/datasets/10kDiabetes.csv')
fwrite(dat, '~/datasets/10kDiabetes_three_text.csv')