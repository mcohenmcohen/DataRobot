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
library(digest)

######################################################
# Download data
######################################################
# https://datarobot.atlassian.net/wiki/spaces/QA/pages/111691866/Release+MBTests

old_release <- 'old'
new_release <- 'new'
test <- 'Eureqa'
TIME_SERIES <- FALSE

if(TIME_SERIES){
  old_mbtest_ids <- '5e4436296246275f29ea4d40'  # eur customer only
  new_mbtest_ids <- '5e4418522f064e7dd7741090'  # eur customer only
} else {
  old_mbtest_ids <- '5e43327c2aa79f6bd6f2f79a'  # current with preds
  new_mbtest_ids <- '5e43328b2aa79f6bd6f2f9e8'  # current with preds
}

prefix = 'http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests='
suffix = '&max_sample_size_only=false'

old_mbtest_urls <- paste0(prefix, old_mbtest_ids, suffix)
new_mbtest_urls <- paste0(prefix, new_mbtest_ids, suffix)

download_data <- function(id){
  out <- fread(paste0(prefix, id, suffix))
  set(out, j='mbtestid', value=id)
}

dat_old_raw <- pblapply(old_mbtest_ids, download_data)
dat_new_raw <- pblapply(new_mbtest_ids, download_data)

######################################################
# Convert possible int64s to numeric
######################################################
dat_old <- copy(dat_old_raw)
dat_new <- copy(dat_new_raw)

clean_data <- function(x){
  x[,Max_RAM_GB := as.numeric(Max_RAM / 1e9)]
  x[,Total_Time_P1_Hours := as.numeric(Total_Time_P1 / 3600)]
  x[,size_GB := as.numeric(size / 1e9)]
  x[,dataset_size_GB := as.numeric(dataset_size / 1e9)]
  return(x)
}

dat_old <- lapply(dat_old, clean_data)
dat_new <- lapply(dat_new, clean_data)

stopifnot(all(sapply(dat_old, function(x) 'Max_RAM_GB' %in% names(x))))
stopifnot(all(sapply(dat_new, function(x) 'Max_RAM_GB' %in% names(x))))

######################################################
# Combine data
######################################################

get_names <- function(x){
  not_int64 <- sapply(x,  class) != 'integer64'
  names(x)[not_int64]
}

names_old <- Reduce(intersect, lapply(dat_old, get_names))
names_new <- Reduce(intersect, lapply(dat_new, get_names))
names_all <- intersect(names_new, names_old)

stopifnot('Metablueprint' %in% names_all)

# The old and new releases. 
dat_old <- lapply(dat_old, function(x) x[,names_all,with=F])
dat_new <- lapply(dat_new, function(x) x[,names_all,with=F])

dat_old <- rbindlist(dat_old, use.names=T)
dat_new <- rbindlist(dat_new, use.names=T)

dat_old <- dat_old[!grepl("Z", dat_old$training_length),]
dat_new <- dat_new[!grepl("Z", dat_new$training_length),]

dat_old[,run := old_release]
dat_new[,run := new_release]

stopifnot(dat_old[,all(Metablueprint=='Metablueprint v12.0.03-so' | Metablueprint=='')])
stopifnot(dat_new[,all(Metablueprint=='Metablueprint v12.0.03-so' | Metablueprint=='')])

dat <- rbindlist(list(dat_old, dat_new), use.names=T)

######################################################
# Exclude some rows
######################################################

dat <- dat[is_prime == FALSE,]
dat <- dat[is_blender == FALSE,]

######################################################
# Remove dupes
######################################################

fastTime <- function(x){
  uniq <- sort(unique(x))
  map <- match(x, uniq)
  uniq <- as.POSIXct(uniq)
  return(uniq[map])
}
dat[,Project_Date := fastTime(Project_Date)]
dat[,key := stri_paste(run, Filename, dataset_yaml_hash, sep=' ')]
dat[,dup := duplicated(key, fromLast = T) | duplicated(key, fromLast = F)]

#Inspect
dat[,table(dup)]
dat[Filename == 'image_flower.csv', list(run, Project_Date, LogLoss_H)]

dat[,keep := (!dup) | (Project_Date == max(Project_Date)), by=key]
dat <- dat[which(keep),]

#Inspect
dat[,table(dup)]
dat[Filename == 'image_flower.csv', list(run, Project_Date, LogLoss_H)]

# Cleanup
dat[,c('key', 'dup', 'keep') := NULL]

# dat[Filename == "openML_datasets_micro-mass_v1.csv", Filename := "micro-mass_v1.csv"]
# dat[Filename == "test_data_jpReview-music-class.csv", Filename := "jpReview-music-class.csv"]

######################################################
# Sanitize names for validation/holdout/prediction data
######################################################

common_cleanup <- function(x){
  x <- gsub('Gini Norm', 'Gini', x, fixed=T)
  x <- gsub('Tweedie Deviance', 'Tweedie', x, fixed=T)
  x <- gsub(' ', '_', x, fixed=T)
  x <- gsub('-', '_', x, fixed=T)
  return(x)
}

validation_metrics <- names(dat)[grepl('_P1$', names(dat))]
validation_metrics_sanitized <- gsub('_P1$', '_V', validation_metrics, )
validation_metrics_sanitized <- common_cleanup(validation_metrics_sanitized)

holdout_metrics <- names(dat)[grepl('_H$', names(dat))]
holdout_metrics_sanitized <- common_cleanup(holdout_metrics)

prediction_metrics <- names(dat)[grepl('^Prediction', names(dat))]
prediction_metrics_sanitized <- gsub('Prediction ', '', prediction_metrics)
prediction_metrics_sanitized <- paste0(prediction_metrics_sanitized, '_P')
prediction_metrics_sanitized <- common_cleanup(prediction_metrics_sanitized)

setnames(dat, validation_metrics, validation_metrics_sanitized)
setnames(dat, holdout_metrics, holdout_metrics_sanitized)
setnames(dat, prediction_metrics, prediction_metrics_sanitized)

######################################################
# Cleanup metrics
######################################################

# Identify metrics we have for validation/holdout/prediciton
all_metrics <- list(
  gsub('_V$', '', validation_metrics_sanitized),
  gsub('_H$', '', holdout_metrics_sanitized),
  gsub('_P$', '', prediction_metrics_sanitized)
)
all_metrics <- Reduce(intersect, all_metrics)

# Convert to numeric
for(metric in all_metrics){  # Loop through all metrics
  for(set in c('_V', '_H', '_P')){  # Loop through validation/holdout/prediction sets
    
    var = paste0(metric, set)
    
    # Pull the raw data which needs to be cleabed
    dirty_data = dat[[var]]
    
    # Print points that will get converted to NA
    tmp = sort(unique(dirty_data))
    wont_convert = !is.finite(as.numeric(tmp))
    if(any(wont_convert)){
      print(tmp[wont_convert])
    }
    
    # Convert dirt data to clean data
    set(dat, j=var, value=as.numeric(dirty_data))
  }
}

######################################################
# Add some vars
######################################################

dat[,dataset_bin := cut(dataset_size_GB, unique(c(0, 1.5, 2.5, 5, ceiling(max(dataset_size_GB)))), ordered_result=T, include.lowest=T)]
dat[,sample_round := Sample_Pct]
dat[sample_round=='--', sample_round := '0']
dat[,sample_round := round(as.numeric(sample_round))]

# Drop non-time series data
if(TIME_SERIES){
  dat <- dat[Sample_Pct == '--',]
} else{
  dat <- dat[Sample_Pct != '--',]
}

######################################################
# Summarize stats - numeric 
######################################################

# NOTE: sample_round == NA will be ALL sample sizes across the whole AutoPilot

res <- copy(dat)

byvars <- list(
  c('run', 'Filename', 'dataset_yaml_hash', 'Y_Type', 'sample_round', 'main_task'),
  c('run', 'Filename', 'dataset_yaml_hash', 'Y_Type', 'sample_round'),
  c('run', 'Filename', 'dataset_yaml_hash', 'Y_Type')
)
res_num <- lapply(byvars, function(x){
  res[,list(
    Max_RAM_GB = max(Max_RAM_GB, na.rm=T),
    Total_Time_P1_Hours = max(Total_Time_P1_Hours, na.rm=T),
    Gini_V = max(Gini_V, na.rm=T),
    Gini_H = max(Gini_H, na.rm=T),
    Gini_P = max(Gini_P, na.rm=T),
    RMSE_V = min(RMSE_V, na.rm=T),
    RMSE_H = min(RMSE_H, na.rm=T),
    RMSE_P = min(RMSE_P, na.rm=T),
    LogLoss_V = min(LogLoss_V, na.rm=T),
    LogLoss_H = min(LogLoss_H, na.rm=T),
    LogLoss_P = min(LogLoss_P, na.rm=T),
    Tweedie_V = min(Tweedie_V, na.rm=T),
    Tweedie_H = min(Tweedie_H, na.rm=T),
    Tweedie_P = min(Tweedie_P, na.rm=T),
    MASE_V = min(MASE_V, na.rm=T),
    MASE_H = min(MASE_H, na.rm=T),
    MASE_P = min(MASE_P, na.rm=T)
  ), by=x]
})
res_num <- rbindlist(res_num, fill=T, use.names = T)

res_num = melt.data.table(res_num, id.vars=byvars[[1]])
res_num = dcast.data.table(
  res_num, 
  Filename + dataset_yaml_hash + Y_Type + sample_round + main_task + variable ~ run, 
  value.var='value')

set(res_num, j='diff', value = res_num[[new_release]] - res_num[[old_release]])
set(res_num, j='pct_diff', value = res_num[[new_release]] / res_num[[old_release]] - 1)

######################################################
# Summarize stats - best model by metric (for RAM/runtime it's the worst model)
######################################################

# NOTE: sample_round == NA will be ALL sample sizes across the whole AutoPilot

res[,main_task_copy := main_task]  # make a copy of main_task, as main_task[ selection won't work when we aggregate by main task
res_cat <- lapply(byvars, function(x) {
  res[,list(
    Max_RAM_GB = main_task_copy[which.max(Max_RAM_GB)],
    Total_Time_P1_Hours = main_task_copy[which.max(Total_Time_P1_Hours)],
    Gini_V = main_task_copy[which.max(Gini_V)],
    Gini_H = main_task_copy[which.max(Gini_H)],
    Gini_P = main_task_copy[which.max(Gini_P)],
    RMSE_V = main_task_copy[which.min(RMSE_V)],
    RMSE_H = main_task_copy[which.min(RMSE_H)],
    RMSE_P = main_task_copy[which.min(RMSE_P)],
    LogLoss_V = main_task_copy[which.min(LogLoss_V)],
    LogLoss_H = main_task_copy[which.min(LogLoss_H)],
    LogLoss_P = main_task_copy[which.min(LogLoss_P)],
    Tweedie_V = main_task_copy[which.min(Tweedie_V)],
    Tweedie_H = main_task_copy[which.min(Tweedie_H)],
    Tweedie_P = main_task_copy[which.min(Tweedie_P)],
    MASE_V = main_task_copy[which.min(MASE_V)],
    MASE_H = main_task_copy[which.min(MASE_H)],
    MASE_P = main_task_copy[which.min(MASE_P)]
  ), by=x]
})
res_cat <- rbindlist(res_cat, fill=T, use.names = T)


res_cat[,run := paste0(run, '_best_model')]
res_cat = melt.data.table(res_cat, id.vars=byvars[[1]])
res_cat = dcast.data.table(
  res_cat, 
  Filename + dataset_yaml_hash + Y_Type + sample_round + main_task + variable ~ run, 
  value.var='value')

res = merge(res_cat, res_num, by=intersect(names(res_cat), names(res_num)), all=TRUE)

######################################################
# Find last autopilot stage
######################################################

# todo: check files with final_autopilot_stage != 3
res_ranks = res[!is.na(sample_round ),list(sample_round=sort(unique(sample_round))), by=c('Filename', 'dataset_yaml_hash')]
res_ranks[,autopilot_stage := rank(sample_round), by=c('Filename', 'dataset_yaml_hash')]
res_ranks[,final_autopilot_stage := max(autopilot_stage[sample_round < 80]), by=c('Filename', 'dataset_yaml_hash')]
res_ranks[,table(final_autopilot_stage)]

res = merge(res, res_ranks, by=c('Filename', 'dataset_yaml_hash', 'sample_round'), all.x=T)

######################################################
# Plot of results
######################################################
plot_dat = res

if(TIME_SERIES){
  plot_vars = c('Max_RAM_GB', 'Total_Time_P1_Hours', 'LogLoss_V', 'MASE_V')
  plot_name = 'time series results'
  
} else{
  plot_vars = c('Max_RAM_GB', 'Total_Time_P1_Hours', 'LogLoss_H', 'Gini_H')
  plot_name = 'non time series results'
}
plot_dat = plot_dat[variable %in% plot_vars & !is.na(diff),]

# NA means "grand total"
dat_overall = plot_dat[is.na(main_task) & is.na(sample_round),]
dat_stage = plot_dat[is.na(main_task) & !is.na(sample_round),]
dat_model = plot_dat[!is.na(main_task) & !is.na(sample_round),]

# Project level
ggplot(dat_overall, aes_string(x=old_release, y=new_release, color='Y_Type')) + 
  geom_point() + geom_abline(slope=1, intercept=0) + 
  facet_wrap(~variable, scales='free') + 
  theme_bw() + theme_tufte() + ggtitle(paste(test, old_release, 'vs', new_release, plot_name, 'project level'))

# # Autopilot stage level
# ggplot(dat_stage, aes_string(x=old_release, y=new_release, color='Y_Type')) + 
#   geom_point() + geom_abline(slope=1, intercept=0) +
#   facet_wrap(~variable, scales='free') + 
#   theme_bw() + theme_tufte() + ggtitle(paste(test, old_release, 'vs', new_release, plot_name, 'autopilot stage level'))

# Model level
ggplot(dat_model, aes_string(x=old_release, y=new_release, color='Y_Type')) + 
  geom_point() + geom_abline(slope=1, intercept=0) +
  facet_wrap(~variable, scales='free') + 
  theme_bw() + theme_tufte() + ggtitle(paste(test, old_release, 'vs', new_release, plot_name, 'model level'))

######################################################
# Lookit issues - project level
######################################################

res_table = copy(dat_model)  # dat_overall or dat_stage or dat_model
res_table[,dataset_yaml_hash := NULL]
res_table[,autopilot_stage := NULL]
res_table[,final_autopilot_stage := NULL]
#res_table[,main_task := NULL]

# Accuracy changes
accuracy_vars = c(
  'Gini_V', 'Gini_H', 'Gini_P', 
  'LogLoss_V', 'LogLoss_H', 'LogLoss_P',
  'RMSE_V', 'RMSE_H', 'RMSE_P',
  'MASE_V', 'MASE_H', 'MASE_P'
)
res_table[variable %in% accuracy_vars & abs(pct_diff)>0.1,][order(variable, pct_diff),]

# Runtime changes
res_table[variable %in% 'Total_Time_P1_Hours' & abs(diff)>0.10 & abs(pct_diff)>0.10,][order(Filename, main_task, sample_round),]

# RAM changes
res_table[variable %in% 'Max_RAM_GB' & abs(diff)>0.25 & abs(pct_diff)>0.25,][order(Filename, main_task, sample_round),]


######################################################
# sumamry stats
######################################################

summary(res_table[variable == 'MASE_V',])
summary(res_table[variable == 'MASE_H',])

summary(res_table[variable == 'Gini_V',])
summary(res_table[variable == 'Gini_H',])

summary(res_table[variable == 'LogLoss_V',])
summary(res_table[variable == 'LogLoss_H',])
