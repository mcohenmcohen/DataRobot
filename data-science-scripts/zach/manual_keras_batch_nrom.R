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
# https://datarobot.atlassian.net/wiki/spaces/QA/pages/111691866/Release+MBTests

old_mbtest_ids <- c(
  '5dd4357779f075000dc3da56'
)
new_mbtest_ids <- c(
  '5dd2c7c6e92ae2000da685ce'
)

old_release <- 'master'
new_release <- 'new_batch_norm'
test <- 'Docker'

prefix = 'http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests='
suffix = '&max_sample_size_only=false'

old_mbtest_urls <- paste0(prefix, old_mbtest_ids, suffix)
new_mbtest_urls <- paste0(prefix, new_mbtest_ids, suffix)

dat_old_raw <- pblapply(old_mbtest_urls, fread)
dat_new_raw <- pblapply(new_mbtest_urls, fread)

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
# Clean Data
######################################################

get_names <- function(x){
  not_int64 <- sapply(x,  class) != 'integer64'
  names(x)[not_int64]
}

names_old <- Reduce(intersect, lapply(dat_old, get_names))
names_new <- Reduce(intersect, lapply(dat_new, get_names))
names_all <- intersect(names_new, names_old)

stopifnot('Metablueprint' %in% names_all)

dat_old <- lapply(dat_old, function(x) x[,names_all,with=F])
dat_new <- lapply(dat_new, function(x) x[,names_all,with=F])

dat_old <- rbindlist(dat_old, use.names=T)
dat_new <- rbindlist(dat_new, use.names=T)

dat_old <- dat_old[!grepl("Z", dat_old$training_length),]
dat_new <- dat_new[!grepl("Z", dat_new$training_length),]

dat_old[,run := old_release]
dat_new[,run := new_release]

stopifnot(dat_old[,all(Metablueprint=='Test_Keras v2')])
stopifnot(dat_new[,all(Metablueprint=='Test_Keras v2')])

######################################################
# Combine data
######################################################

dat <- rbindlist(list(dat_old, dat_new), use.names=T, fill=T)

######################################################
# Add some vars
######################################################

dat[,dataset_bin := cut(dataset_size_GB, unique(c(0, 1.5, 2.5, 5, ceiling(max(dataset_size_GB)))), ordered_result=T, include.lowest=T)]
dat[,sample_round := Sample_Pct]
dat[sample_round=='--', sample_round := '0']
dat[,sample_round := round(as.numeric(sample_round))]

######################################################
# Exclude some rows
######################################################

dat <- dat[is_prime == FALSE,]

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

dat[,key := stri_paste(run, Filename, sep=' ')]
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

######################################################
# Summarize stats
######################################################

res <- copy(dat)
res <- res[!is.na(Max_RAM_GB),]
res <- res[!is.na(Total_Time_P1_Hours),]
res <- res[!is.na(`Gini Norm_H`),]

# CLASSIFICATION ONLY
res <- res[Y_Type %in% c('Binary', 'Multiclass'),]

# Lookup vars
# names(dat)[grepl('auc', tolower(names(dat)))]

# Convert to numeric
measures = c(
  'Max_RAM_GB', 'Total_Time_P1_Hours', 
  'Gini Norm_P1', 'Gini Norm_H', 'Prediction Gini Norm', 
  'AUC_P1', 'AUC_H', 'Prediction AUC',
  'LogLoss_P1', 'LogLoss_H', 'Prediction LogLoss'
  )
for(v in measures){
  tmp = sort(unique(res[[v]]))
  wont_convert = !is.finite(as.numeric(tmp))
  if(any(wont_convert)){
    print(tmp[wont_convert])
  }
  set(res, j=v, value=as.numeric(res[[v]]))
}

# Aggregate
res <- res[,list(
  Max_RAM_GB = max(Max_RAM_GB, na.rm=T),
  Total_Time_P1_Hours = max(Total_Time_P1_Hours, na.rm=T),
  Gini_V = max(`Gini Norm_P1`, na.rm=T),
  Gini_H = max(`Gini Norm_H`, na.rm=T),
  Gini_P = max(`Prediction Gini Norm`, na.rm=T),
  AUC_V = max(AUC_P1, na.rm=T),
  AUC_H = max(AUC_H, na.rm=T),
  AUC_P = max(`Prediction AUC`, na.rm=T),
  Logloss_V = min(`LogLoss_P1`, na.rm=T),
  Logloss_H = min(`LogLoss_H`, na.rm=T),
  Logloss_P = min(`Prediction LogLoss`, na.rm=T)
), by=c('run', 'Filename', 'Y_Type', 'sample_round')]

measures = c(
  'Max_RAM_GB', 'Total_Time_P1_Hours', 
  'Gini_V', 'Gini_H', 'Gini_P', 
  'AUC_V', 'AUC_H', 'AUC_P', 
  'Logloss_V', 'Logloss_H', 'Logloss_P')
for(v in measures){
  tmp = sort(unique(res[[v]]))
  wont_convert = !is.finite(as.numeric(tmp))
  if(any(wont_convert)){
    print(tmp[wont_convert])
  }
  set(res, j=v, value=as.numeric(res[[v]]))
}
res = melt.data.table(res, measure.vars=intersect(names(res), measures))
res = res[is.finite(value),]
res = dcast.data.table(res, Filename + Y_Type + sample_round + variable ~ run, value.var='value')   # + main_task

set(res, j='diff', value = res[[new_release]] - res[[old_release]])
# res[abs(diff) < 0.01 & variable == 'Gini_H' & (sample_round == 0 | sample_round == 64),]

######################################################
# Plot of results
######################################################

plot_vars = c('Max_RAM_GB', 'Total_Time_P1_Hours', 'Logloss_H', 'AUC_H')
ggplot(res[sample_round == 64 & variable %in% plot_vars & !is.na(diff),], aes_string(x=old_release, y=new_release, color='Y_Type')) + 
  geom_point() + geom_abline(slope=1, intercept=0) + 
  facet_wrap(~variable, scales='free') + 
  theme_bw() + theme_tufte() + ggtitle(paste(test, old_release, 'vs', new_release, 'non time series results'))

######################################################
# Table of results - non multiclass
######################################################

DEC=4
res[!is.na(diff),list(
  min=round(min(diff), DEC),
  mean=round(mean(diff), DEC),
  sd=round(sd(diff), DEC),
  median=round(median(diff), DEC),
  max=round(max(diff), DEC),
  pct_higher=sum(diff>0)/.N,
  .N
), by=variable][order(variable),]

DEC=4
res[!is.na(diff),list(
  min=round(min(diff), DEC),
  mean=round(mean(diff), DEC),
  sd=round(sd(diff), DEC),
  median=round(median(diff), DEC),
  max=round(max(diff), DEC),
  pct_higher=sum(diff>0)/.N,
  .N
), by=c('Y_Type', 'variable')][order(Y_Type, variable),]
