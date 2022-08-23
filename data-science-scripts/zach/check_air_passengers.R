a = data.table::fread('https://s3.amazonaws.com/datarobot_public_datasets/time_series/AirPassengers.csv')
a = a[['y']]
b = as.numeric(datasets::AirPassengers)
all.equal(a, b)
