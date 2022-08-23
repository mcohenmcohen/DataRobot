library(yaml)
datasets <- yaml.load_file('~/workspace/mbtest-datasets/mbtest_datasets/data/Functionality/OTP/mbtest_data_otp_pure_time_series_with_preds.yaml')
datasets <- lapply(datasets, function(x){
  out <- list()
  out[['dataset_name']] = x[['dataset_name']]
  out[['prediction_dataset_name']] = x[['prediction_dataset_name']]
  out[['target']] = x[['target']]
  out[['partitioning']] = list(
    partition_column = x[['partitioning']][['partition_column']]
  )
  out
})
datasets[[78]]
datasets <- as.yaml(datasets)
on <- '~/workspace/mbtest-datasets/mbtest_datasets/data/Functionality/OTP/mbtest_data_otp_default_partitioning_test.yaml'
cat(c(
  "# Yaml for pure time series with all defaults",
  "# Default metric",
  "# Default paritioning",
  "# Default backtests",
  "# Initially used to test MMSQUAD-1746",
  "# Should be new default time series yaml when Mbtest passes",
  "# https://datarobot.atlassian.net/browse/MMSQUAD-1746",
  ""
), file = on, sep='\n', append=F)
cat(datasets, file = on, sep='\n', append=T)
system(paste('head -n 25', on))
