library('data.table')
dat <- fread('https://s3.amazonaws.com/datarobot_public_datasets/Heller_Motor_claim_count_80.csv')
dat[,sum(is.na(claim_count))]

dat <- fread('https://s3.amazonaws.com/datarobot_public_datasets/Heller_Motor_cost_80.csv')
dat[,sum(is.na(cost))]

dat[,sum(cost==0)/.N]

