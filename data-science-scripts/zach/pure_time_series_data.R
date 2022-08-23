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
  'PT1H' = 1440,
  'PT20H' = 24,
  'PT20M' = 60,
  'P140D' = 52,
  'P20M' = 12,
  'PT3H' = 10, #Special case
  'P20D' = 7,
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
  } else if(grepl('https://s3.amazonaws.com/datarobot_public_datasets/time_series/SP500', x[['dataset_name']], fixed=T)){
    duration <- 'P30D'
    x[['partitioning']][['validation_duration']] <- duration
    x[['partitioning']][['holdout_duration']] <- duration
  }else if(grepl('https://s3.amazonaws.com/datarobot_public_datasets/time_series/calcium', x[['dataset_name']], fixed=T)){
    duration <- 'PT1H'
    x[['partitioning']][['validation_duration']] <- duration
    x[['partitioning']][['holdout_duration']] <- duration
  }else if(grepl('https://s3.amazonaws.com/datarobot_public_datasets/time_series/methane-input-into-gas-furnace', x[['dataset_name']], fixed=T)){
    duration <- 'PT3H'
    x[['partitioning']][['validation_duration']] <- duration
    x[['partitioning']][['holdout_duration']] <- duration
  }else if(grepl('https://s3.amazonaws.com/datarobot_public_datasets/time_series/occupancy_dataset_train.csv', x[['dataset_name']], fixed=T)){
    duration <- 'PT6H'
    x[['partitioning']][['validation_duration']] <- duration
    x[['partitioning']][['holdout_duration']] <- duration
  }
  return(x)
}

###############################################################################
# Built in datasets to R
###############################################################################

if(RE_DOWNLOAD_DATA){
  # AirPassengers
  data(AirPassengers)
  data(co2)
  ts_to_dr(AirPassengers, '1949-01-01', '1960-12-31', 'month', '~/datasets/time_series/AirPassengers.csv')
  #ts_to_dr(co2, '1959-01-01', '1997-12-31', 'month', '~/datasets/time_series/maunaloa_co2.csv')

  weekly_co2 <- fread('ftp://aftp.cmdl.noaa.gov/products/trends/co2/co2_weekly_mlo.txt', fill=T, skip=50)
  weekly_co2[,date := as.Date(ISOdate(V1, V2, V3))]
  weekly_co2[,y := V5]
  weekly_co2 <- weekly_co2[y>0,]
  weekly_co2 <- weekly_co2[,list(date, y)]
  write_csv(weekly_co2, '~/datasets/time_series/weekly_earth_co2.csv')

  monthly_co2 <- fread('ftp://aftp.cmdl.noaa.gov/products/trends/co2/co2_mm_mlo.txt', fill=T, skip=72)
  monthly_co2[,date := as.Date(ISOdate(V1, V2, 1))]
  monthly_co2[,y := V4]
  monthly_co2 <- monthly_co2[y>0,]
  monthly_co2 <- monthly_co2[,list(date, y)]
  write_csv(monthly_co2, '~/datasets/time_series/monthly_earth_co2.csv')
}

###############################################################################
# Rob Hyndman's website
###############################################################################

if(RE_DOWNLOAD_DATA){
  #https://robjhyndman.com/publications/complex-seasonality/

  # (a) U.S. finished motor gasoline products supplied
  # (thousands of barrels per day),
  # weekly data from February 1991 to July 2005.
  gas <- read.csv("https://robjhyndman.com/data/gasoline.csv")[,1]
  gas <- ts(gas, start=1991+31/365.25, frequency = 365.25/7)
  ts_to_dr(gas, '1991-01-20', '2005-04-17', 'week', '~/datasets/time_series/hyndman_usa_gasoline.csv')

  # (b) Number of calls handled on weekdays between 7:00 am and 9:05 pm
  # Five-minute call volume from March 3, 2003, to May 23, 2003
  # in a large North American commercial bank.
  # calls <- unlist(read.csv("https://robjhyndman.com/data/callcenter.txt",
  #                          header=TRUE,sep="\t")[,-1])
  # calls <- msts(calls, start=1, seasonal.periods = c(169, 169*5))
  calls <- fread("https://robjhyndman.com/data/callcenter.txt")
  setnames(calls, 'V1', 'time')
  calls <- melt.data.table(calls, variable.name='date', value.name='y')
  calls[,date := as.Date(date, '%d/%m/%Y')]
  calls[,date := as.POSIXct(paste(date, time))]
  calls <- calls[,list(date, y)]
  calls[,date := as.character(date)]
  write_csv(calls, '~/datasets/hyndman_bank_call_center_volume.csv')

  # (c) Turkish electricity demand data.
  # Daily data from 1 January 2000 to 31 December 2008.
  telec <- fread("https://robjhyndman.com/data/turkey_elec.csv", header=F)
  ts_to_dr(telec[['V1']], '2000-01-01', '2008-12-31', 'day', '~/datasets/time_series/hyndman_turkish_electricity_demand.csv')
}

###############################################################################
# Rob Hyndman's Forecasting, principles and practice
###############################################################################

#TOO SHORT:
#ts_to_dr(ausair, '1970-01-01', '2009-01-01', 'year', '~/datasets/time_series/fpp_autralian_airpassengers.csv')
#ts_to_dr(austa, '1980-01-01', '2010-09-01', 'year', '~/datasets/time_series/fpp_autralian_visitors.csv')
#ts_to_dr(austourists, '1999-01-01', '2010-12-01', 'year', '~/datasets/time_series/fpp_autralian_tourists.csv')
#ts_to_dr(cafe, '1982-04-01', '2010-12-01', 'quarter', '~/datasets/time_series/fpp_autralian_tourists.csv')
#elecsales
#euretail
#insurance
#livestock
#oil
#sunspotarea
#vn
#wmurders

#JUST RIGHT
if(RE_DOWNLOAD_DATA){
  ts_to_dr(a10, '1991-07-01', '2008-06-01', 'month', '~/datasets/time_series/fpp_antidiabetic_drugs.csv')
  ts_to_dr(ausbeer, '1956-01-01', '2008-09-01', 'quarter', '~/datasets/time_series/fpp_autralian_beer.csv')
  ts_to_dr(debitcards, '2000-01-01', '2012-12-01', 'month', '~/datasets/time_series/fpp_iceland_debit_card_usage.csv')
  ts_to_dr(departures[,1], '1976-01-01', '2012-02-01', 'month', '~/datasets/time_series/fpp_aus_departures_permanent.csv')
  ts_to_dr(departures[,2], '1976-01-01', '2012-02-01', 'month', '~/datasets/time_series/fpp_aus_departures_reslong.csv')
  ts_to_dr(departures[,3], '1976-01-01', '2012-02-01', 'month', '~/datasets/time_series/fpp_aus_departures_vislong.csv')
  ts_to_dr(departures[,4], '1976-01-01', '2012-02-01', 'month', '~/datasets/time_series/fpp_aus_departures_resshort.csv')
  ts_to_dr(departures[,5], '1976-01-01', '2012-02-01', 'month', '~/datasets/time_series/fpp_aus_departures_visshort.csv')
  ts_to_dr(elecequip, '1996-01-01', '2011-11-01', 'month', '~/datasets/time_series/fpp_eu_electric_equipment.csv')
  ts_to_dr(h02, '1991-07-01', '2008-06-01', 'month', '~/datasets/time_series/fpp_aus_cortecosteroid_sales.csv')
  ts_to_dr(melsyd[,1], '1987-06-01', '1992-10-26', 'week', '~/datasets/time_series/fpp_aus_air_traffic_first.csv')
  ts_to_dr(melsyd[,2], '1987-06-01', '1992-10-26', 'week', '~/datasets/time_series/fpp_aus_air_traffic_business.csv')
  ts_to_dr(melsyd[,3], '1987-06-01', '1992-10-26', 'week', '~/datasets/time_series/fpp_aus_air_traffic_economy.csv')
  ts_to_dr(usconsumption[,1], '1970-01-01', '2010-12-01', 'quarter', '~/datasets/time_series/fpp_us_personal_consumption_change.csv')
  ts_to_dr(usconsumption[,2], '1970-01-01', '2010-12-01', 'quarter', '~/datasets/time_series/fpp_us_personal_income_change.csv')
  ts_to_dr(usmelec, '1973-01-01', '2010-10-01', 'month', '~/datasets/time_series/fpp_us_electricity_net_generation.csv')
}

###############################################################################
# Twitter's anomaly detection
###############################################################################

#https://github.com/twitter/AnomalyDetection
library(AnomalyDetection)
if(RE_DOWNLOAD_DATA){
  data(raw_data)
  raw_data$timestamp <- as.POSIXct(raw_data$timestamp)
  raw_data <- as.data.table(raw_data)
  raw_data[,date := as.character(timestamp)]
  raw_data <- raw_data[,list(date, y=count)]
  write_csv(raw_data, '~/datasets/time_series/twitter_volume_over_time.csv')
}

###############################################################################
# Facebook's prohpet
###############################################################################

if(RE_DOWNLOAD_DATA){
  # Datasets from facebook
  #https://github.com/facebookincubator/prophet
  dat_fb <- fread('https://raw.githubusercontent.com/facebookincubator/prophet/master/R/tests/testthat/data.csv')
  dat_retail <- fread('https://raw.githubusercontent.com/facebookincubator/prophet/master/examples/example_retail_sales.csv')
  dat_wp <- fread('https://raw.githubusercontent.com/facebookincubator/prophet/master/examples/example_wp_R.csv')
  dat_wp_outliers1 <- fread('https://raw.githubusercontent.com/facebookincubator/prophet/master/examples/example_wp_R_outliers1.csv')
  dat_wp_outliers2 <- fread('https://raw.githubusercontent.com/facebookincubator/prophet/master/examples/example_wp_R_outliers2.csv')
  dat_pm <- fread('https://raw.githubusercontent.com/facebookincubator/prophet/master/examples/example_wp_peyton_manning.csv')

  for(dt in list(dat_fb, dat_retail, dat_wp, dat_wp_outliers1, dat_wp_outliers2, dat_pm)){
    setnames(dt, 'ds', 'date')
  }

  write_csv(dat_fb, '~/datasets/time_series/facebook_data.csv')
  write_csv(dat_retail, '~/datasets/time_series/facebook_retail.csv')
  write_csv(dat_wp, '~/datasets/time_series/facebook_wp_no_outliers.csv')
  write_csv(dat_wp_outliers1, '~/datasets/time_series/facebook_wp_with_outliers1.csv')
  write_csv(dat_wp_outliers2, '~/datasets/time_series/facebook_wp_with_outliers2.csv')
  write_csv(dat_pm, '~/datasets/time_series/facebook_peyton_manning.csv')
  # plot(dat_fb$y, type='l')
  # plot(dat_retail$y, type='l')
  # plot(dat_wp$y, type='l')
  # plot(dat_wp_outliers1$y, type='l')
  # plot(dat_wp_outliers2$y, type='l')
  # plot(dat_pm$y, type='l')
}

###############################################################################
# ISO New England
###############################################################################

if(RE_DOWNLOAD_DATA){
  #https://www.iso-ne.com/isoexpress/web/reports/load-and-demand/-/reports/dmnd-rt-hourly-sys?p_auth=6zuc30WN
  right_now <- format(Sys.Date(), '%Y%m%d')
  iso_ne_raw <- fread(paste0('https://www.iso-ne.com/transform/csv/hourlysystemdemand?start=20080701&end=', right_now), fill=T, skip=5, drop='H')

  iso_ne <- copy(iso_ne_raw)
  iso_ne <- iso_ne[!is.na(MWh),]
  iso_ne[,Date := as.Date(Date, format='%m/%d/%Y')]
  iso_ne[,Date := as.character(Date)]
  iso_ne <- iso_ne[,list(date=Date, HE, y=MWh)]
  write_csv(iso_ne, '~/datasets/time_series/iso_new_hourly_load.csv')
}

###############################################################################
# Reformat ISO New England
###############################################################################

#unlink('~/datasets/time_series/iso_new_hourly_load.csv')
dat <- fread(
  'https://s3.amazonaws.com/datarobot_public_datasets/time_series/iso_new_hourly_load.csv',
  colClasses=c(HE='character'))
dat[,date := as.Date(date)]
dat[,HE_num := as.integer(HE)]
dat[,HE := paste0('H', HE)]
dat[HE == '02X',HE_num := 3]
dat[,datetime := ISOdatetime(year(date), month(date), mday(date), HE_num-1, 0, 0)]
dat[HE == '02X',datetime := datetime - 60*60]
#dat[,datetime := format(datetime, usetz=TRUE)]
#dat[,datetime := format(datetime, tz="America/New_York", usetz=TRUE)]
#dat[,datetime := format(datetime-4*60*60, tz="UTC")]
dat[,datetime := format(datetime, tz="America/New_York", usetz=FALSE)]
# dat[date==as.Date('2008-11-02'),]
# dat[date==as.Date('2009-03-08'),]
# dat[date==as.Date('2017-01-01'),]
# dat[date==as.Date('2017-06-01'),]
dat <- dat[,list(date=date, HE, y)]
dat[,table(HE)]
write_csv(dat, '~/datasets/time_series/iso_ne_hourly_load.csv')

###############################################################################
# Generate yaml file - no preds
###############################################################################

#Generate yaml
DIR <- '~/datasets/time_series/'
S3_DIR <- 'https://s3.amazonaws.com/datarobot_public_datasets/time_series/'
datasets <- sort(list.files(DIR))
datasets <- setdiff(datasets, c('train', 'test')) #Folders
all_data <- pblapply(datasets, function(x){
  dat <- fread(paste0(DIR, x))
  stopifnot(c('date', 'y') %in% names(dat))
  out <- list(
    dataset_name = paste0(S3_DIR, x),
    target = 'y',
    metric = 'RMSE',
    partitioning = list(
      partition_column = 'date',
      autopilot_data_selection_method = 'rowCount'
    )
  )
  if(nrow(dat) > 200){
    out[['partitioning']][['number_of_backtests']] <- 2L
  }
  if(nrow(dat) > 1000){
    out[['partitioning']][['number_of_backtests']] <- 5L
  }
  out
})
names(all_data) <- datasets

#Convert to yaml and save
names(all_data) <- NULL
yaml_out <- as.yaml(lapply(all_data, add_duration))

note <- "# Pure time series data for Mbtesting and manual testing
# Generated from: https://github.com/datarobot/data-science-scripts/blob/master/zach/pure_time_series_data.R

# Sources:
# https://www.rdocumentation.org/packages/datasets/versions/3.4.0/topics/AirPassengers
# ftp://aftp.cmdl.noaa.gov/products/trends/co2/co2_weekly_mlo.txt
# ftp://aftp.cmdl.noaa.gov/products/trends/co2/co2_mm_mlo.txt
# https://robjhyndman.com/publications/complex-seasonality/
# https://www.rdocumentation.org/packages/fpp/versions/0.5
# https://github.com/twitter/AnomalyDetection
# https://github.com/facebookincubator/prophet
# https://www.iso-ne.com/isoexpress/web/reports/load-and-demand/

"

outfile <- '~/workspace/mbtest-datasets/mbtest_datasets/data/Functionality/OTP/mbtest_data_otp_pure_time_series_no_preds.yaml'
cat(note, file=outfile, append=F)
cat(yaml_out, file=outfile, append=T)
#system(paste('head -n 25', outfile))

###############################################################################
# Generate yaml file - preds
###############################################################################

#http://shrink.prod.hq.datarobot.com/mbtests/594d194439b1bc109e02e44d

#Fix: fpp_us_personal_consumption_change_train - 72 M

#Fix: fpp_us_personal_income_change_train
# Traceback (most recent call last):
#   File "/opt/shrink/mbtests/project_runner.py", line 241, in run
# self._api_select_target()
# File "/opt/shrink/mbtests/project_runner.py", line 1779, in _api_select_target
# self._api_project.set_target(**kwargs)
# File "/usr/local/lib/python2.7/site-packages/admin_public_api/models/project.py", line 118, in set_target
# return super(AdminProject, self).set_target(target, metric=metric, **set_target_args)
# File "/usr/local/lib/python2.7/site-packages/datarobot/models/project.py", line 962, in set_target
# self.from_async(async_location, max_wait=max_wait)
# File "/usr/local/lib/python2.7/site-packages/datarobot/models/project.py", line 604, in from_async
# max_wait=max_wait)
# File "/usr/local/lib/python2.7/site-packages/datarobot/async.py", line 50, in wait_for_async_resolution
# raise errors.AsyncProcessUnsuccessfulError(e_template.format(data))
# AsyncProcessUnsuccessfulError: The job did not complete successfully. Job Data: {u'status': u'ERROR', u'message': u'A minimum of 100 rows is required in the largest backtest fold but only found 94 rows.', u'code': 1, u'created': u'2017-06-23T17:08:48.654930Z'}

#Generate yaml
DIR <- '~/datasets/time_series/'
S3_DIR <- 'https://s3.amazonaws.com/datarobot_public_datasets/time_series/'
datasets <- sort(list.files(DIR))
datasets <- setdiff(datasets, c('train', 'test')) #Folders
last <- 'AirPassengers.csv'
datasets <- c(setdiff(datasets, last), last)
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
  dat <- fread(paste0(DIR, filename))

  #Clean names
  setnames(dat, make.names(tolower(names(dat)), unique=TRUE))

  #Set correct column order
  stopifnot(c('date', 'y') %in% names(dat))
  dat <- dat[order(as.integer(as.POSIXct(date))),]

  #Check dates
  if(any(duplicated(dat[['date']]))){
    warning('Only univariate time series supported at the moment')
  }

  #Try splitting TVH
  fq <- detect_freq(dat)
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
  return(out)
})

#Convert to yaml
names(all_data) <- NULL
yaml_out <- as.yaml(all_data)

#Save
outfile <- '~/workspace/mbtest-datasets/mbtest_datasets/data/Functionality/OTP/mbtest_data_otp_pure_time_series_with_preds.yaml'
cat(note, file=outfile, append=F)
cat('\n', file=outfile, append=T)
cat(yaml_out, file=outfile, append=T)
#system(paste('head -n 75', outfile))

###############################################################################
# Pre-process wunderground
###############################################################################

DIR <- '~/datasets/cfds_time_series/'
weather <- read_csv(paste0(DIR, 'other/wunderground.csv'))
weather <- as.data.table(weather)
targets <- names(weather)[grepl('actual', names(weather))]
locations <- weather[,sort(unique(location))]
xvars <- setdiff(names(weather), c(targets, 'location'))
for(target in targets){
  for(l in locations){
    dat <- weather[location==l,]
    dat <- dat[,c(target, xvars),with=FALSE]
    setnames(dat, target, 'y')
  }
  write_csv(dat, paste0(DIR, 'wunderground_', l, '_', target, '.csv'))
  write_csv(dat, paste0( '~/datasets/upload/wunderground_', l, '_', target, '.csv'))
}

###############################################################################
# Fix gdelt
###############################################################################

gdelt <- fread('~/datasets/cfds_time_series/other/gdelt_wti_daily.csv')
gdelt[,y := as.numeric(y)]
gdelt <- gdelt[!is.na(y),]
write_csv(gdelt, '~/datasets/cfds_time_series/gdelt_wti_daily.csv')
write_csv(gdelt, '~/datasets/upload/occupancy_dataset.csv')

###############################################################################
# Add prefix to Greg's S&P 500 data
###############################################################################

DIR <- '~/datasets/cfds_time_series/S&P/'
SP500 <- pblapply(list.files(DIR), function(x){
  out <- fread(paste0(DIR, x))
  if(nrow(out) > 175){
    write_csv(out, paste0('~/datasets/cfds_time_series/SP500_', x))
    write_csv(out, paste0('~/datasets/upload/SP500_', x))
  }
  return(NULL)
})

###############################################################################
# Fix João's M1/M3 comp data
###############################################################################

DIR <- '~/datasets/cfds_time_series/M1M3/'
M1M3 <- pblapply(list.files(DIR), function(x){
  out <- fread(paste0(DIR, x))
  out[,partition := NULL]
  setnames(out, tolower(names(out)))
  write_csv(out, paste0('~/datasets/cfds_time_series/', x))
  write_csv(out, paste0('~/datasets/upload/', x))
  return(NULL)
})

###############################################################################
# Make a copy of the large yaml
###############################################################################

newfile <- gsub(
  'mbtest_data_otp_pure_time_series_with_preds.yaml',
  'mbtest_data_otp_pure_time_series_with_preds_huge.yaml',
  outfile)
system(paste('cp', outfile, newfile))

###############################################################################
# Add CFDS datasets to yaml
###############################################################################

#Generate yaml
DIR <- '~/datasets/cfds_time_series/'
S3_DIR <- 'https://s3.amazonaws.com/datarobot_public_datasets/time_series/'
datasets <- sort(list.files(DIR))
datasets <- setdiff(datasets, c('train', 'test', 'S&P', 'M1M3', 'other')) #Folders
SP500 <- datasets[grepl('^SP500', datasets)]
MCOMP <- datasets[grepl('^N', datasets)]
CALCIUM <- datasets[grepl('^calcium', datasets)]
#datasets <- setdiff(datasets, CALCIUM) #For now

#Sample 10 each from SP500 and Mcomp
set.seed(42)
SP500_5 <- sample(SP500, 5)
MCOMP_5 <- sample(MCOMP, 5)
CALCIUM_5 <- sample(CALCIUM, 5)
CALCIUM_5 <- 'calcium_3_10.csv'
MCOMP_25 <- c(MCOMP_5, sample(MCOMP, 20))
DROP <- c(setdiff(SP500, SP500_5), setdiff(MCOMP, MCOMP_25), setdiff(CALCIUM, CALCIUM_5))

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

#Remove some S&P 500 + M3 comp datasets from the full file
all_data_small <- copy(all_data)
for(d in DROP){
  all_data_small[[d]] <- NULL
}

print(length(all_data))
print(length(all_data_small))

#Convert to yaml
names(all_data) <- NULL
names(all_data_small) <- NULL
yaml_out_small <- as.yaml(all_data_small)
yaml_out_large <- as.yaml(all_data)

#Save small
cat('\n', file=outfile, append=T)
cat('#CFDS Datasets:\n', file=outfile, append=T)
cat(yaml_out_small, file=outfile, append=T)
#system(paste('head -n 75', outfile))

#Save large
cat('\n', file=newfile, append=T)
cat('#CFDS Datasets:\n', file=newfile, append=T)
cat(yaml_out_large, file=newfile, append=T)
#system(paste('head -n 75', outfile))

###############################################################################
# Dataset number
###############################################################################
print(length(yaml.load_file(outfile)))
print(length(yaml.load_file(newfile)))
