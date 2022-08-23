######################################################
# Setup
######################################################

stop()
rm(list=ls(all=T))
gc(reset=T)
library(pbapply)
library(data.table)
library(bit64)
library(ggplot2)
library(Hmisc)
library(jsonlite)
library(reshape2)
library(stringi)
library(ggplot2)
library(ggthemes)

######################################################
# Download data
######################################################

# Newer set of master Mbtests
mbtest_ids <- c(
  '5cc357377347c90029a46165' # Current with preds Yaml
)

prefix = 'http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests='
suffix = '&max_sample_size_only=false'

mbtest_ids <- paste0(prefix, mbtest_ids, suffix)
dat_raw <- pblapply(mbtest_ids, fread)

######################################################
# Convert possible int64s to numeric
######################################################

dat <- copy(dat_raw)

clean_data <- function(x){
  x[,Max_RAM_GB := as.numeric(Max_RAM / 1e9)]
  x[,Total_Time_P1_Hours := as.numeric(Total_Time_P1 / 3600)]
  x[,Total_Time_P1_Seconds := as.numeric(Total_Time_P1)]
  x[,size_GB := as.numeric(size / 1e9)]
  x[,dataset_size_GB := as.numeric(dataset_size / 1e9)]
  x[,x_prod_2_max_cardinal := NULL]
  return(x)
}
dat <- lapply(dat, clean_data)

######################################################
# Combine data within each test
######################################################

get_names <- function(x){
  not_int64 <- sapply(x,  class) != 'integer64'
  names(x)[not_int64]
}

names_all <- Reduce(intersect, lapply(dat, get_names))

stopifnot('Metablueprint' %in% names_all)

dat <- lapply(dat, function(x) x[,names_all,with=F])
dat <- rbindlist(dat, use.names=T)

stopifnot(dat[,all(Metablueprint=='Metablueprint v12.0.03-so')])

######################################################
# Add some vars
######################################################

dat[,dataset_bin := cut(dataset_size_GB, unique(c(0, 1.5, 2.5, 5, ceiling(max(dataset_size_GB)))), ordered_result=T, include.lowest=T)]
dat[,sample_round := Sample_Pct]
dat[sample_round=='--', sample_round := '0']
dat[,sample_round := round(as.numeric(sample_round))]

dat[,Filename := gsub('.csv', '', Filename, fixed=T)]

######################################################
# Summarize stats - non multiclass
######################################################

# Add one variable
dat[,`Prediction MSE` := (`Prediction RMSE`) ^ 2]

measures = c(
  'Total_Time_P1_Seconds',
  'error_H',
  'Prediction LogLoss',
  'Prediction AUC',
  'Prediction MAD',
  'Prediction MSE',
  'Prediction RMSE',
  'Prediction R Squared')

# Convert measures to numeric
for(v in measures){
  tmp = sort(unique(dat[[v]]))
  wont_convert = !is.finite(as.numeric(tmp))
  if(any(wont_convert)){
    print(v)
    print(tmp[wont_convert])
  }
  set(dat, j=v, value=as.numeric(dat[[v]]))
}

# Aggregate by autopilot stage
# Assumption 1: The "time" for an autopilot stage is the slowest model.  All other models run in parallel
# Assumption 2: Total runtime is the sum of the runtime for each autopilot stage
agg_time <- dat[,list(
  Approx_Training_Seconds = max(Total_Time_P1_Seconds)
), by=c('Filename', 'sample_round')]
agg_time <- agg_time[,list(
  Approx_Training_Seconds = sum(Approx_Training_Seconds)
), by='Filename']

# Aggregate by project
agg <- dat[,list(
  min_error_H = min(error_H, na.rm=T),
  max_error_H = max(error_H, na.rm=T),
  log_loss = min(`Prediction LogLoss`, na.rm=T),
  roc_auc_score = max(`Prediction AUC`, na.rm=T),
  mean_absolute_error = min(`Prediction MAD`, na.rm=T),
  mean_squared_error = min(`Prediction MSE`, na.rm=T),
  rmse = min(`Prediction RMSE`, na.rm=T),
  r2_score = max(`Prediction R Squared`, na.rm=T)
), by=c('Filename', 'metric')]
agg[,error_H := min_error_H]
agg[metric %in% c('Accuracy', 'AUC', 'Balanced Accuracy'), error_H := max_error_H]
agg[,c('min_error_H', 'max_error_H') := NULL]

# Join 2 aggs
agg <- merge(agg, agg_time, by='Filename', all=T)

# Reorder columns
agg <- agg[,list(
  Filename,
  Approx_Training_Seconds,
  error_H,
  metric,
  log_loss,
  roc_auc_score,
  mean_absolute_error,
  mean_squared_error,
  rmse,
  r2_score
)]

# Subset to datasets in the test
test_files <- c(
  'allstate_reg_subset_1095605_80',
  'census_1990_full_80',
  'fastiron-train-full_80',
  'msd_full_80',
  'allstate_classif_subset_1095605_80',
  'bloggers_small_80',
  'chargeback_clean_80',
  'check_transform',
  'digits-train_80',
  'ohsumed_binary_80',
  'reuters_earnVacq_80',
  'sentiment140_twitter_1txt_100k_80'
)
for(f in test_files){
  if(! f %in% agg[,Filename]){
    print(f)
  }
}
agg <- agg[Filename %in% test_files, ]
fwrite(agg, '~/datasets/datarobot_results.csv')
agg