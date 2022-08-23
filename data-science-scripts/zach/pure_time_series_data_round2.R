stop()
rm(list=ls(all=T))
gc(reset=T)

###############################################################################
# Setup
###############################################################################

#TODO: rename args so they match all the way through the function calls

library(forecast)
library(fpp)
library(data.table)
library(reshape2)
library(readr)
library(pbapply)
library(yaml)
library(parallel)
library(lubridate)

RE_DOWNLOAD_DATA <- FALSE

ts_to_dr <- function(x, start, end, by, filename){
  dt <- data.table(
    date = seq.Date(as.Date(start), as.Date(end), by),
    y = as.numeric(x)
  )
  dt[,date := as.character(date)]
  dt <- dt[!is.na(y),]
  print(dt)
  stopifnot(nrow(dt) > 140)
  write_csv(dt, filename)
}

detect_freq <- function(dat){
  dt <- dat[,as.integer(as.POSIXct(date))]
  dt <- min(diff(unique(dt))) / (3600*24)

  if(dt <=0){
    stop('Data are not correctly order')
  }

  #Special case for iso ne
  if('HE' %in% names(dat)){
    return(24)
  }

  #9-second special case
  if(dt == 0.00625){
    return(10)
  }

  #Secondly
  if(dt <= 1.157407e-05 * 1.10){ #1 second + wiggle room
    return(60) #Seconds in a minute
  }

  #Minutely
  if(dt < 10/(24*60)){ #1/(24*60) + wiggle room
    return(1440) #1440 minutes in a day
  }

  #Hourly
  if(dt < 1/12){ #1/24 + wiggle room
    return(24) #24 hours in a day
  }

  #Daily
  if(dt < 2){ #1 + wiggle room
    return(7) #7 days in a week
  }

  #Weekly
  if(dt < 9){ #7 + wiggle room
    return(52) #52 weeks in a year
  }

  #Monthly
  if(dt < 40){ #30 + wiggle room
    return(12) #12 months in a year
  }

  #Quarterly
  if(dt < 180){ #150ish + wiggle room
    return(4) #4 quarters in a year
  }

  #Yearly ¯\_(ツ)_/¯
  return(1) #No seasonality
}

split_t_v_h <- function(x, fq = detect_freq(x)){

  time <- x[,as.integer(as.POSIXct(date))]
  x_local <- copy(x[order(time),])

  ten_pct <- nrow(x_local) * .10
  close_to_ten_pct <- round(ten_pct/fq) * fq
  close_to_ten_pct <- max(close_to_ten_pct, fq)

  pt_h <- nrow(x_local) - close_to_ten_pct
  pt_v <- pt_h - close_to_ten_pct

  out <- list(
    t = x_local[1:pt_v,],
    v = x_local[(pt_v+1):pt_h,],
    h = x_local[(pt_h+1):nrow(x_local),]
  )

  stopifnot(all(rbindlist(out) == x_local))

  return(out)
}

#Modify < 200 row datasets
freq_map <- c(
  'PT20M' = 1440,
  'PT20H' = 24,
  'PT20M' = 60,
  'P140D' = 52,
  'P20M' = 12,
  'PT198S' = 10, #Special case
  'P20W' = 7,
  'P6Y' = 4,
  'P20Y' = 1)
add_duration <- function(x, train=fread(x[['dataset_name']])){
  #For validation sets <= 1 day, or < 200 rows, we need to setup ourselves
  duration <- train[,as.POSIXct(unique(date))]
  duration <- as.integer(ceiling(diff(range(duration))))
  if(nrow(train) < 200 | duration <= 5){
    freq <- detect_freq(train)
    duration <- names(freq_map)[freq_map==freq]
    x[['partitioning']][['validation_duration']] <- duration
    x[['partitioning']][['holdout_duration']] <- duration
  } else if(grepl('^SP500', x[['dataset_name']])){
    duration <- 'P30D' #SP500 has gaps, so use 30 day test sets
    x[['partitioning']][['validation_duration']] <- duration
    x[['partitioning']][['holdout_duration']] <- duration
  }
  return(x)
}

###############################################################################
# Add CFDS datasets to yaml
###############################################################################

#Generate yaml
DIR <- '~/datasets/cfds_time_series_r2/'
S3_DIR <- 'https://s3.amazonaws.com/datarobot_public_datasets/time_series/'
datasets <- sort(list.files(DIR))
datasets <- setdiff(datasets, c('train', 'test', 'S&P', 'wunderground', 'electricity')) #Folders
SP500 <- datasets[grepl('^SP500', datasets)]
MCOMP <- datasets[grepl('^N', datasets)]
CALCIUM <- datasets[grepl('^calcium', datasets)]
#datasets <- setdiff(datasets, CALCIUM) #For now

#Iterate through CFDS data
all_data <- pblapply(datasets, FUN = function(filename){ #

  library(forecast)
  library(fpp)
  library(data.table)
  library(reshape2)
  library(readr)
  library(pbapply)
  library(yaml)
  library(parallel)

  #Start by assuming we'll make the TVH split
  print(filename)
  ds_name <- gsub('.csv', '', filename, fixed=T)
  out <- list(
    dataset_name = paste0(S3_DIR, ds_name, '_train.csv'),
    prediction_dataset_name = paste0(S3_DIR, ds_name, '_test.csv'),
    target = 'y',
    metric = 'RMSE',
    partitioning = list(
      partition_column = 'date',
      autopilot_data_selection_method = 'rowCount'
    )
  )

  #Load Data
  #print(filename)
  dat <- fread(paste0(DIR, filename))

  #Clean names
  setnames(dat, make.names(tolower(names(dat)), unique=TRUE))

  #Set correct column order
  first <- c('date', 'y')
  stopifnot(first %in% names(dat))
  dat <- dat[order(as.integer(as.POSIXct(date))),]
  setcolorder(dat, c(first, setdiff(names(dat), first)))
  dat <- dat[!is.na(y),]
  dat <- dat[!is.na(date),]

  #Check dates
  if(any(duplicated(dat[['date']]))){
    stop('Only univariate time series supported at the moment')
  }

  #Try splitting TVH
  fq <- detect_freq(dat)
  if(fq> nrow(dat)){
    fq <- 1
  }
  tvh <- split_t_v_h(dat, fq)
  tv <- rbind(tvh[['t']], tvh[['v']])

  #If data is too short, re-split with a frequency of 1
  if(nrow(tv) < 140){
    tvh <- split_t_v_h(dat, fq=1)
    tv <- rbind(tvh[['t']], tvh[['v']])
  }

  #If it's still too short, no TVH split
  #Use full dataset for training
  #Use no predictions
  if(nrow(tv) < 140){
    out[['dataset_name']] <- paste0(S3_DIR, ds_name, '.csv')
    out[['prediction_dataset_name']] <- NULL
  }

  #Save TVH split to upload to S3
  train_file <- paste0('~/datasets/upload/', ds_name, '_train.csv')
  test_file <- paste0('~/datasets/upload/', ds_name, '_test.csv')
  write_csv(tv, train_file)
  write_csv(tvh[['h']], test_file)

  #Add duration
  out <- add_duration(out, tv)

  #Add backtests
  if(nrow(dat) > 500){
    out[['partitioning']][['number_of_backtests']] <- 2L
  }
  if(nrow(dat) > 1000){
    out[['partitioning']][['number_of_backtests']] <- 5L
  }

  #Return
  out <- add_duration(out, tv)
  return(out)
})
names(all_data) <- datasets
all_data[[1]]

#Convert to yaml
names(all_data) <- NULL
yaml_out <- as.yaml(all_data)

#Save small
outfile <- '~/workspace/mbtest-datasets/mbtest_datasets/data/Functionality/OTP/mbtest_data_otp_pure_time_series_with_preds.yaml'
cat('\n', file=outfile, append=T)
cat(yaml_out, file=outfile, append=T)

#Save large
newfile <- gsub(
  'mbtest_data_otp_pure_time_series_with_preds.yaml',
  'mbtest_data_otp_pure_time_series_with_preds_huge.yaml',
  outfile)
cat('\n', file=newfile, append=T)
cat(yaml_out, file=newfile, append=T)

###############################################################################
# Dataset number
###############################################################################
print(length(yaml.load_file(outfile)))
print(length(yaml.load_file(newfile)))
