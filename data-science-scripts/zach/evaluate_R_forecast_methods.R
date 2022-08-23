stop()
rm(list=ls(all=T))

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
library(pbapply)
library(yaml)
library(parallel)
library(ggplot2)
library(prophet)
library(bit64)

#Todo: add prophet to forecast
#Todo: compare to DR
#Todo: run on the huge yaml

normalizedGini <- function(aa, pp) {
  Gini <- function(a, p) {
    if (length(a) !=  length(p)) stop("Actual and Predicted need to be equal lengths!")
    temp.df <- data.frame(actual = a, pred = p, range=c(1:length(a)))
    temp.df <- temp.df[order(-temp.df$pred, temp.df$range),]
    population.delta <- 1 / length(a)
    total.losses <- sum(a)
    null.losses <- rep(population.delta, length(a)) # Hopefully is similar to accumulatedPopulationPercentageSum
    accum.losses <- temp.df$actual / total.losses # Hopefully is similar to accumulatedLossPercentageSum
    gini.sum <- cumsum(accum.losses - null.losses) # Not sure if this is having the same effect or not
    sum(gini.sum) / length(a)
  }
  Gini(aa,pp) / Gini(aa,aa)
}

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

  #Minutely
  if(dt < 10/(24*60)){ #1/24 + wiggle room
    return(1440) #1440 minutes in a day
  }

  #Hourly
  if(dt < 1/12){ #1/24 + wiggle room
    return(24) #24 hours in a day
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

forecast_compare <- function(x_raw, y_raw, freq, method=ets, plot=FALSE, title=title, ...){
  x <- x_raw[order(date),y]
  y <- y_raw[order(date),y]


  N <- max(1000, length(y), freq*10)
  if(length(x) > N){
    x <- tail(x, N)
  }

  x <- ts(x, freq=freq, start=1)
  y <- ts(y, freq=freq, start=length(x)/freq+1)
  model <- method(x, ...)
  fcast <- forecast(model, length(y))
  if(plot){
    plot(fcast)
    title(sub=title)
    lines(y, col='red')
  }
  stopifnot(length(fcast$mean) == length(y))
  out <- data.table(accuracy(fcast, y))
  out <- out[2,]
  out[, Gini_Norm := normalizedGini(as.numeric(y), fcast$mean)]
  out[,method := capture.output(print(method))[1]]
  first <- 'method'
  setcolorder(out, c(first, setdiff(names(out), first)))
  return(out)
}

split_t_v_h <- function(x, fq = detect_freq(x)){

  time <- x[,as.integer(as.POSIXct(date))]
  x_local <- copy(x[order(time),])

  ten_pct <- nrow(x) * .10
  close_to_ten_pct <- round(ten_pct/fq) * fq
  close_to_ten_pct <- max(close_to_ten_pct, fq)

  pt_h <- nrow(x) - close_to_ten_pct
  pt_v <- pt_h - close_to_ten_pct

  out <- list(
    t = x[1:pt_v,],
    v = x[(pt_v+1):pt_h,],
    h = x[(pt_h+1):nrow(x),]
  )

  stopifnot(all(rbindlist(out) == x))

  return(out)
}

ets_ <- function(x, ...) ets(x, ...)
arima_ <- function(x, ...) auto.arima(x, stepwise=FALSE, parallel=TRUE, num.cores=32)
tbats_ <- function(x, ...) tbats(x, seasonal.periods = frequency(x), num.cores=32, use.parallel=TRUE, ...)

wrappers <- list(
  function(x) ets_(x),
  function(x) arima_(x),
  function(x) tbats_(x)
)

best_forecast_rmse <- function(tvh, fq, title, plot=T){
  models <- lapply(wrappers, function(x) forecast_compare(tvh[['t']], tvh[['v']], fq, x))
  best <- which.min(sapply(models, '[[', 'RMSE'))
  res <- forecast_compare(rbind(tvh[['t']], tvh[['v']], fill=T), tvh[['h']], fq, wrappers[[best]], plot=plot, title=title)
  return(res)
}

#Modify < 200 row datasets
freq_map <- c(
  'PT20M' = 1440,
  'PT20H' = 24,
  'P140D' = 52,
  'P20M' = 12,
  'P80M' = 4,
  'P20Y' = 1)
add_duration <- function(x){
  train <- fread(x[['dataset_name']])
  if(nrow(train) < 200){
    freq <- detect_freq(train)
    duration <- names(freq_map)[freq_map==freq]
    x[['partitioning']][['validation_duration']] <- duration
    x[['partitioning']][['holdout_duration']] <- duration
  }
  return(x)
}

###############################################################################
# Iterate through yaml and predict with forecast package
###############################################################################

set.seed(42)
datasets <- yaml.load_file('~/workspace/mbtest-datasets/mbtest_datasets/data/Functionality/OTP/mbtest_data_otp_pure_time_series_with_preds.yaml')
datasets <- lapply(datasets, function(x){
  if(! is.null(x$prediction_dataset_name)){
    return(x)
  }
  return(NULL)
})
datasets <- datasets[!sapply(datasets, is.null)]
length(datasets)
#datasets <- sample(datasets, 10)

eval <- function(x, y, n){
  x <- as.numeric(x)
  y <- as.numeric(y)
  data.table(
    dataset = n,
    `Prediction RMSE` = sqrt(mean((x-y)^2)),
    `Prediction MAD` = median(abs((x-y))),
    `Prediction Gini Norm` = normalizedGini(y, x)
  )
}

evaluate_model <- function(x, FUN, ...){
  library(data.table)
  library(forecast)
  library(prophet)
  library(bit64)
  #print(dput(x))
  print(x$dataset_name)
  if(! is.null(x$prediction_dataset_name)){

    #Download data
    tryCatch({
      train <- fread(utils::URLencode(x$dataset_name))
      if(is.integer64(train$y)){
        train[,y := as.numeric(y)]
      }
      train <- train[order(as.integer(as.POSIXct(date))),]
      fq <- detect_freq(train)
      ts <- train[,ts(y, frequency = fq)]
      # if(length(ts)>1000){
      #   t <- tail(ts, 1000)
      # }

      #Fit model
      mod <- FUN(ts, ...)

      #Download test set
      test <- fread(utils::URLencode(x$prediction_dataset_name))
      test <- test[order(as.integer(as.POSIXct(date))),]

      #Evaluate
      fcast <- forecast(mod, nrow(test))
      return(eval(fcast$mean, test$y, x$dataset_name))
    }, error = function(e){
      warning(e)
      return(NULL)
    })
  }
  return(NULL)
}

#ets
cl <- makePSOCKcluster(70, outfile='')
clusterExport(cl, ls())
ets_res_raw <- pblapply(datasets, evaluate_model, ets_, cl=cl)
ets_res <- rbindlist(ets_res_raw)
ets_res[,dataset := gsub('https://s3.amazonaws.com/datarobot_public_datasets/time_series/', '', dataset, fixed=T)]
ets_res[,method := 'Ets']
stopCluster(cl)

#auto.arima
cl <- makePSOCKcluster(60, outfile='')
arima_res_raw <- pblapply(datasets, evaluate_model, arima_, cl=cl)
arima_res <- rbindlist(arima_res_raw)
arima_res[,dataset := gsub('https://s3.amazonaws.com/datarobot_public_datasets/time_series/', '', dataset, fixed=T)]
arima_res[,method := 'Arima']
stopCluster(cl)

#tbats
tbats_res_raw <- pblapply(datasets, evaluate_model, tbats_)
tbats_res <- rbindlist(tbats_res_raw)
tbats_res[,dataset := gsub('https://s3.amazonaws.com/datarobot_public_datasets/time_series/', '', dataset, fixed=T)]
tbats_res[,method := 'Tbats']

#prophet
cl <- makePSOCKcluster(60, outfile='')
prophet_res_raw <- pblapply(datasets, function(x){
  library(data.table)
  library(forecast)
  library(prophet)
  library(bit64)
  if(! is.null(x$prediction_dataset_name)){
    tryCatch({
      train <- fread(utils::URLencode(x$dataset_name))
      if(is.integer64(train$y)){
        train[,y := as.numeric(y)]
      }
      train <- train[,ds := as.POSIXct(date)]
      train <- train[,list(ds, y)]
      train <- train[order(-ds)]

      test <- fread(utils::URLencode(x$prediction_dataset_name))
      test <- test[,ds := as.POSIXct(date)]
      test <- test[,list(ds, y)]
      test <- test[order(ds)]

      mod <- prophet(train)
      pred <- predict(mod, test)
      pred <- pred[order(pred$ds),]
      return(eval(pred$yhat, test$y, x$dataset_name))
    }, error = function(e){
      warning(e)
      return(NULL)
    })
  }
  return(NULL)
})
prophet_res <- rbindlist(prophet_res_raw)
prophet_res[,dataset := gsub('https://s3.amazonaws.com/datarobot_public_datasets/time_series/', '', dataset, fixed=T)]
prophet_res[,method := 'prophet']
stopCluster(cl)

#Shut down cluster
stopCluster(cl)

###############################################################################
# Grab datarobot results
###############################################################################
#http://shrink.prod.hq.datarobot.com/mbtests/594d194439b1bc109e02e44d

#Old, possibly incomplete tests
#DR_raw <- fread('http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests=594d194439b1bc109e02e44d&max_sample_size_only=false')
#DR_raw <- fread('http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests=595ff626200fee10b2c2a7c7&max_sample_size_only=false')

#RMSE
DR_raw <- fread('http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests=59662d068d97703f1d509799&max_sample_size_only=false')

#DR choses metric
#DR_raw <- fread('http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests=5967c73425c7e52d54f90f34&max_sample_size_only=false')

#Analyze runtime for Mike
time <- DR_raw[!is.na(Total_Time_P1),list(
  time_min = min(Total_Time_P1),
  time_25pt = quantile(Total_Time_P1, .25),
  time_med = median(Total_Time_P1),
  time_mean = mean(Total_Time_P1),
  time_75pt = quantile(Total_Time_P1, .75),
  time_max = max(Total_Time_P1),
  time_sd = sd(Total_Time_P1)
), by='Filename'][order(time_max, time_med, time_min),]
write_csv(time, '~/datasets/pure_time_series_runtime.csv')

dput(names(DR_raw)[grepl('Gini', names(DR_raw))])
dput(names(DR_raw)[grepl('RMSE', names(DR_raw))])
dput(names(DR_raw)[grepl('MAD', names(DR_raw))])
dput(names(DR_raw)[grepl('Prediction', names(DR_raw))])
dput(names(DR_raw)[grepl('Sample', names(DR_raw))])

DR <- copy(DR_raw)
DR <- DR[!is.na(`Prediction Gini Norm`), ]
DR[,dataset := Filename]
DR[,method := 'DataRobot']
#DR[,best_gini := as.integer(1:.N == which.max(`Gini Norm_H` *.75 + `Gini Norm_P1` *.25)), by='dataset']
DR[,best_gini := as.integer(1:.N == which.max(`Gini Norm_H`)), by='dataset']
DR[,best_rmse := as.integer(1:.N == which.min(`RMSE_H`)), by='dataset']
DR[,best_mad := as.integer(1:.N == which.min(`MAD_H`)), by='dataset']

keep <- c('dataset', 'method', 'Prediction RMSE', 'Prediction MAD', 'Prediction Gini Norm')
DR_gini <- DR[best_gini == 1,][,keep,with=F]
DR_rmse <- DR[best_rmse == 1,][,keep,with=F]
DR_mad <- DR[best_mad == 1,][,keep,with=F]

ggplot(DR[best_gini == 1,], aes(x=`Gini Norm_H`, y=`Prediction Gini Norm`)) +
  geom_point() + theme_bw() + geom_abline(slope=1, intercept=0) +
  ggtitle('DataRobot Holdout set vs. Prediction Set Gini Norm')

###############################################################################
# Save for report
###############################################################################

FC <- rbindlist(list(ets_res, arima_res, tbats_res), fill=T)
PR <- prophet_res
save(DR_gini, DR_rmse, DR_mad, FC, PR, file='~/workspace/data-science-scripts/zach/compare_data.Rdata')

write_csv(FC, '~/datasets/forecast_results.csv')
write_csv(PR, '~/datasets/prophet_results.csv')
