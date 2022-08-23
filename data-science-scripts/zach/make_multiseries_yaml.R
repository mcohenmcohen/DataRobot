rm(list=ls(all=TRUE))
gc(reset=TRUE)
library(yaml)
library(httr)

# Multi
multiseries <- list.files('~/datasets/cfds_multiseries/uploaded/')
multiseries <- lapply(multiseries, function(x){
  list(
    dataset_name=paste0('https://s3.amazonaws.com/datarobot_public_datasets/', x),
    target='y',
    use_time_series=TRUE,
    partitioning=list(partition_column='date'),
    time_series=list(multi_series_id_columns=list('id'))
  )
})
check = lapply(multiseries, function(x){stop_for_status(HEAD(x[['dataset_name']]))})
OUTFILE <- "~/workspace/mbtest-datasets/mbtest_datasets/data/Functionality/TimeSeries/multi_series_without_preds.yaml"
note <- "# Multiseries data for Mbtesting and manual testing
# Generated from: https://github.com/datarobot/data-science-scripts/blob/master/zach/make_multiseries_yaml.R
# Sources: CFDS Team\n\n"
cat(note, file=OUTFILE, append=FALSE)
cat(as.yaml(multiseries), file=OUTFILE, append=TRUE)
system(paste('head -n50', OUTFILE))

# Hour
hourly <- list.files('~/datasets/cfds_hourly/uploaded/')
hourly <- lapply(hourly, function(x){
  list(
    dataset_name=paste0('https://s3.amazonaws.com/datarobot_public_datasets/', x),
    target='y',
    partitioning=list(partition_column='date'),
    use_time_series=TRUE
  )
})
check = lapply(hourly, function(x) stop_for_status(HEAD(x[['dataset_name']])))
OUTFILE <- "~/workspace/mbtest-datasets/mbtest_datasets/data/Functionality/TimeSeries/new_hourly_time_series_without_preds.yaml"
note <- "# Hourly time series data for Mbtesting and manual testing
# Generated from: https://github.com/datarobot/data-science-scripts/blob/master/zach/make_multiseries_yaml.R
# Sources: CFDS Team\n\n"
cat(note, file=OUTFILE, append=FALSE)
cat(as.yaml(hourly), file=OUTFILE, append=TRUE)
system(paste('head -n50', OUTFILE))
