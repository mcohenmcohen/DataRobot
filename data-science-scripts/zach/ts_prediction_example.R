library(data.table)
tr = fread('https://s3.amazonaws.com/datarobot_public_datasets/time_series/hyndman_turkish_electricity_demand_train.csv')
te = fread('https://s3.amazonaws.com/datarobot_public_datasets/time_series/hyndman_turkish_electricity_demand_test.csv')
te[,y := NULL]
tail(tr, 14)
head(te, 7)
