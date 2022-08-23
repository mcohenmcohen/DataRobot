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

validate_yaml <- function(a){

  #Check for weird file types
  check_1 <- grepl('.xls', a$dataset_name, fixed=T)
  check_2 <- grepl('.zip', a$dataset_name, fixed=T)
  check_3 <- a$dataset_name %in% c(
    'https://s3.amazonaws.com/datarobot_public_datasets/stack_overflow_closed_question_1Gb.csv',
    'https://s3.amazonaws.com/datarobot_public_datasets/subreddit_15_classes.csv',
    'https://s3.amazonaws.com/datarobot_public_datasets/fakenewschallenge.csv'
  )

  #Determine the file prefix
  prefix <- 'https://s3.amazonaws.com/datarobot_public_datasets/'
  stopifnot(grepl(prefix, a$dataset_name, fixed=T))

  #URL encode non-prefix
  x <- gsub(prefix, '', a$dataset_name, fixed=T)
  x <- utils::URLencode(x, reserved=T)
  x <- paste0(prefix, x)
  #print(paste0('...', x))

  #Check the size of the file
  out <- HEAD(x)[['headers']][['content-length']]
  out <- as.numeric(out) / (1024 * 1024)
  return(out)
}
######################################################
# Load Yaml
######################################################

INFILE <- "~/workspace/mbtest-datasets/mbtest_datasets/data/Functionality/OTP/mbtest_data_otp_pure_time_series_with_preds.yaml"
OUTFILE <- "~/workspace/mbtest-datasets/mbtest_datasets/data/Functionality/TS/time_series_with_preds.yaml"

yaml <- yaml.load_file(INFILE, handlers=custom_handlers)
yaml <- pblapply(yaml, function(x){
  x[['use_time_series']] <- 'True'
  return(x)
})
sizes <- pbsapply(yaml, validate_yaml)
summary(sizes)
yaml <- yaml[order(sizes)]

######################################################
# Save Yaml
######################################################

note <- "# Pure time series data for Mbtesting and manual testing
# Generated from: https://github.com/datarobot/data-science-scripts/blob/master/zach/convert_otp_to_ts.R

# Sources:
# https://www.rdocumentation.org/packages/datasets/versions/3.4.0/topics/AirPassengers
# ftp://aftp.cmdl.noaa.gov/products/trends/co2/co2_weekly_mlo.txt
# ftp://aftp.cmdl.noaa.gov/products/trends/co2/co2_mm_mlo.txt
# https://robjhyndman.com/publications/complex-seasonality/
# https://www.rdocumentation.org/packages/fpp/versions/0.5
# https://github.com/twitter/AnomalyDetection
# https://github.com/facebookincubator/prophet
# https://www.iso-ne.com/isoexpress/web/reports/load-and-demand/
# BASED ON: "
note <- c(note, INFILE, "

")
cat(note, file=OUTFILE, append=F)
cat(yaml::as.yaml(yaml), file=OUTFILE, append=T)
system(paste('head -n50', OUTFILE))
