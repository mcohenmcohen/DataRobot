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

PREFIX <- 'https://s3.amazonaws.com/datarobot_public_datasets/'
yaml <- pblapply(yaml, function(x){
  set.seed(42)
  dat <- fread(x$dataset_name)
  oldname <- gsub(PREFIX, '', x$dataset_name, fixed=T)
  dat[,weight := runif(.N)]
  x[['metric']] <- 'Weighted LogLoss'
  x[['weights']] <- list(weight = 'weight')
  x[['user_partition_col']] <- 'user_partition'
  x[['training_level']] <- 'T'
  x[['validation_level']] <- 'V'
  x[['holdout_level']] <- 'H'
  x[['multiclass']] <- TRUE
  NEW_PREFIX <- 'weighted_'
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
