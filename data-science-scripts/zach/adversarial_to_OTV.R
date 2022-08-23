######################################################
# Setup
######################################################

stop()
rm(list=ls(all=T))
gc(reset=T)
library(yaml)
library(data.table)
library(readr)
library(httr)
library(pbapply)
#https://stackoverflow.com/a/20921907

rm(list=ls(all=T))
gc(reset=T)
set.seed(42)

handle_nothing <- function(x) x
custom_handlers <- list(
  "bool#y"=handle_nothing,
  "bool#yes"=handle_nothing,
  "bool#T"=handle_nothing,
  "bool#TRUE"=handle_nothing,
  "bool#1"=handle_nothing
)

######################################################
# Load Yaml
######################################################
INFILE <- '~/workspace/mbtest-datasets/mbtest_datasets/data/Functionality/mbtest_data_multiclass_minimum.yaml'
yaml <- yaml.load_file(INFILE, handlers=custom_handlers)

######################################################
# Add random weights and save
######################################################

dates = seq.Date(as.Date('2017-01-01'), as.Date('2017-12-31'), by='day')

PREFIX <- 'https://s3.amazonaws.com/datarobot_public_datasets/'
yaml <- pblapply(yaml, function(x){
  set.seed(42)
  dat <- fread(x$dataset_name)
  oldname <- gsub(PREFIX, '', x$dataset_name, fixed=T)
  dat[,weight := runif(.N)]
  dat[,date := sample(dates, .N, replace=T)]
  x[['metric']] <- 'Weighted LogLoss'
  x[['weights']] <- list(weight = 'weight')
  x[['partitioning']] <- list(
    partition_column = 'date',
    validation_duration = "P0Y3M0DT0H0M0S",
    number_of_backtests = 2,
    autopilot_data_selection_method = 'duration'
  )
  x[['multiclass']] <- TRUE
  NEW_PREFIX <- 'weighted_and_dated_'
  x[['dataset_name']] <- paste0(PREFIX, NEW_PREFIX, oldname)
  newname <- paste0('~/datasets/', NEW_PREFIX, oldname)
  write_csv(dat, newname)
  return(x)
})

######################################################
# Show Yaml to copy/paste into main yaml
######################################################
system('atom ~/workspace/mbtest-datasets/mbtest_datasets/data/Functionality/mbtest_data_multiclass.yaml')
cat(yaml::as.yaml(yaml))
