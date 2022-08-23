library(data.table)
dat_raw = fread('~/datasets/customer/Green_Chef_-_Cloned_Advanced_GLM_Blender_(47+40+44+46+43+45+42+41)_(52)_33.69_ANON_NOdates_Cat_correct.csv')

dat = copy(dat_raw)
dat = dat[!is.na(status_target_wk_02),]

dat[,list(
  act = mean(status_target_wk_02),
  pred = mean(`Cross-Validation Prediction`),
  .N
)]

dat[,list(
  pred = mean(`Cross-Validation Prediction`),
  .N
), by=c('status_target_wk_02')]

dat[,list(
  act = mean(status_target_wk_02),
  pred = mean(`Cross-Validation Prediction`),
  .N
), by='Partition']

set.seed(42)
partition_map = dat[Partition != '-2.0', list(weight=.N), by='Partition']
partition_map[,weight := weight/sum(weight)]
dat[Partition == '-2.0', Partition := sample(partition_map[['Partition']], .N, replace=T, prob=partition_map[['weight']])]

table = dat[,list(
  act = mean(status_target_wk_02),
  pred = mean(`Cross-Validation Prediction`),
  .N
), by='Partition']
table

table[,ratio := (1-pred/act)*100]
table

