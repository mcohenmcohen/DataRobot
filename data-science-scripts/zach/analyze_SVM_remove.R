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
  '5c9a94d67347c900273dbf13'
)
new_mbtest_ids <- c(
  '5c9b7c757347c9002607e546'
)

prefix = 'http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests='
suffix = '&max_sample_size_only=false'

old_mbtest_urls <- paste0(prefix, old_mbtest_ids, suffix)
new_mbtest_urls <- paste0(prefix, new_mbtest_ids, suffix)

dat_old_raw <- pblapply(old_mbtest_urls, fread)
dat_new_raw <- pblapply(new_mbtest_urls, fread)

######################################################
# Convert possible int64s to numeric
######################################################

old_release <- 'Master'
new_release <- 'SVM_Removed'
test <- ''

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

dat_old <- lapply(dat_old, function(x) x[,names_all,with=F])
dat_new <- lapply(dat_new, function(x) x[,names_all,with=F])

dat_old <- rbindlist(dat_old, use.names=T)
dat_new <- rbindlist(dat_new, use.names=T)

dat_old <- dat_old[!grepl("Z", dat_old$training_length),]
dat_new <- dat_new[!grepl("Z", dat_new$training_length),]

dat_old[,run := old_release]
dat_new[,run := new_release]

stopifnot(dat_old[,all(Metablueprint=='Metablueprint v12.0.03-so' | Metablueprint=='')])
stopifnot(dat_new[,all(Metablueprint=='Metablueprint v12.0.03-so')])

dat <- rbindlist(list(dat_old, dat_new), use.names=T)

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
# Summarize ranks
######################################################

res = dat[run=='Master',]
res[,Gini_H := as.numeric(`Gini Norm_H`)]
res <- res[!is.na(Gini_H),]
res[,sample_round := round(sample_round)]
res = res[sample_round %in% c(16, 32, 64, 80),]
res = res[!grepl('BLENDER', main_task),]
res[,main_task := gsub('R$', '', main_task)]
res[,main_task := gsub('C$', '', main_task)]
res[main_task %in% c('ASVME', 'ASVMSK'), main_task := 'SVM']
res[main_task %in% c('SGDRA'), main_task := 'SGD']
res[main_task %in% c('BENETCD2', 'BLENETCD2', 'NB_LENETCD', 'UENETCD', 'ENETCD', 'ENETCDW'), main_task := 'ENET']
res[main_task %in% c('ULENETCD', 'GLMCD', 'LENETCD', 'LENETCDW', 'LR1', 'LRCD'), main_task := 'ENET']
res[main_task %in% c('XL_ENETCD', 'XL_LENETCD'), main_task := 'XLENET']
res[main_task %in% c('ADBLEND', 'ADDMAD', 'ADISOFO'), main_task := 'ANOMALY']
res[main_task %in% c('CNGEC2', 'CNGER2', 'WNGEC2', 'WNGER2'), main_task := 'NGRAM']
res[main_task %in% c('ESXGBC2', 'ESXGBR2', 'RES_PXGBR2', 'RES_XGBR2', 'UESXGBC2', 'UESXGBR2'), main_task := 'XGB']
res[main_task %in% c('XGBR2', 'UPXGBC2', 'UPXGBR2', 'RES_ESXGBR2', 'UXGBC2', 'XGBC2'), main_task := 'XGB']
res[main_task %in% c('PXGBC2', 'PXGBR2', 'UXGBR2'), main_task := 'XGB']
res[main_task %in% c('ESLGBMT', 'RES_ESLGBMT', 'RES_PLGBMT', 'PLGBMT'), main_task := 'LGBM']
res[main_task %in% c('DLNN', 'TFL', 'MLP'), main_task := 'TFNN']
res[sample_round == 80, list(.N), by='main_task'][order(N),]

#res[,sort(table(main_task))]
res = res[,list(
  Gini_H = max(Gini_H)
), by=c('Filename', 'sample_round', 'main_task', 'Blueprint')]
res[,Rank_Gini_H := rank(1-Gini_H), by=c('Filename', 'sample_round')]

# res[,list(
#   min_Rank_Gini_H = min(Rank_Gini_H),
#   mean_Rank_Gini_H = mean(Rank_Gini_H),
#   median_Rank_Gini_H = median(Rank_Gini_H),
#   max_Rank_Gini_H = max(Rank_Gini_H)
# ), by=c('sample_round', 'main_task')]

######################################################
# Summarize stats
######################################################

res <- copy(dat)
res <- res[!is.na(Max_RAM_GB),]
res <- res[!is.na(Total_Time_P1_Hours),]
res <- res[!is.na(`Gini Norm_H`),]

measures = c(
  'Max_RAM_GB', 'Total_Time_P1_Hours', 
  'Gini Norm_P1', 'Gini Norm_H', 'Prediction Gini Norm', 
  'LogLoss_P1', 'LogLoss_H',
  'RMSE_P1', 'RMSE_H',
  'RMSLE_P1', 'RMSLE_H',
  'MASE_P1', 'MASE_H')
for(v in measures){
  stopifnot(v %in% names(res))
  tmp = sort(unique(res[[v]]))
  wont_convert = !is.finite(as.numeric(tmp))
  if(any(wont_convert)){
    print(tmp[wont_convert])
  }
  set(res, j=v, value=as.numeric(res[[v]]))
}

res <- res[,list(
  Max_RAM_GB = max(Max_RAM_GB),
  Total_Time_P1_Hours = max(Total_Time_P1_Hours),
  Gini_V = max(`Gini Norm_P1`),
  Gini_H = max(`Gini Norm_H`),
  Gini_P = max(`Prediction Gini Norm`),
  Gini_H = max(`Gini Norm_H`),
  LogLoss_V = min(`LogLoss_P1`),
  LogLoss_H = min(`LogLoss_H`),
  RMSE_V = min(`RMSE_P1`),
  RMSE_H = min(`RMSE_H`),
  RMSLE_V = min(`RMSLE_P1`),
  RMSLE_H = min(`RMSLE_H`),
  MASE_V = min(`MASE_P1`),
  MASE_H = min(`MASE_H`)
), by=c('run', 'Filename', 'sample_round')]

measures = c(
  'Max_RAM_GB', 'Total_Time_P1_Hours', 
  'Gini_V', 'Gini_H', 'Gini_P', 
  'LogLoss_V', 'LogLoss_H',
  'RMSE_V', 'RMSE_H',
  'RMSLE_V', 'RMSLE_H',
  'MASE_V', 'MASE_H')
res = melt.data.table(res, measure.vars=intersect(names(res), measures))
res = dcast.data.table(res, Filename + sample_round + variable ~ run, value.var='value')

set(res, j='diff', value = res[[new_release]] - res[[old_release]])
# res[abs(diff) < 0.01 & variable == 'Gini_H' & (sample_round == 0 | sample_round == 64),]

######################################################
# Plot of results
######################################################

plot_vars = c('Max_RAM_GB', 'Total_Time_P1_Hours', 'Gini_V', 'Gini_H')
ggplot(res[sample_round == 64 & variable %in% plot_vars & !is.na(diff),], aes_string(x=old_release, y=new_release)) + 
  geom_point() + geom_abline(slope=1, intercept=0) + 
  facet_wrap(~variable, scales='free') + 
  theme_bw() + theme_tufte() + ggtitle(paste(test, old_release, 'vs', new_release, 'non time series results'))
res[variable %in% c('Gini_V', 'Gini_H') & sample_round == 64, summary(diff)]
res[variable %in% c('Gini_H') & sample_round == 80, summary(diff)]

plot_vars = c('Max_RAM_GB', 'Total_Time_P1_Hours', 'MASE_V', 'MASE_H')
ggplot(res[sample_round == 0 & variable %in% plot_vars & !is.na(diff),], aes_string(x=old_release, y=new_release)) + 
  geom_point() + geom_abline(slope=1, intercept=0) + 
  facet_wrap(~variable, scales='free') + 
  theme_bw() + theme_tufte() + ggtitle(paste(test, old_release, 'vs', new_release, 'time series results'))
res[variable %in% c('MASE_V', 'MASE_H') & sample_round == 0, summary(diff)]

plot_vars = c('Max_RAM_GB', 'Total_Time_P1_Hours', 'LogLoss_V', 'LogLoss_H')
ggplot(res[sample_round == 64 & variable %in% plot_vars & !is.na(diff),], aes_string(x=old_release, y=new_release)) + 
  geom_point() + geom_abline(slope=1, intercept=0) + 
  facet_wrap(~variable, scales='free') + 
  theme_bw() + theme_tufte() + ggtitle(paste(test, old_release, 'vs', new_release, 'non time series results'))
res[variable %in% c('LogLoss_V', 'LogLoss_H') & sample_round == 64, summary(diff)]
res[variable %in% c('LogLoss_H') & sample_round == 80, summary(diff)]
res[variable %in% c('LogLoss_H') & sample_round == 80 & diff < -.10,]

plot_vars = c('Max_RAM_GB', 'Total_Time_P1_Hours', 'RMSE_V', 'RMSE_H')
ggplot(res[sample_round == 64 & variable %in% plot_vars & !is.na(diff),], aes_string(x=old_release, y=new_release)) + 
  geom_point() + geom_abline(slope=1, intercept=0) + 
  facet_wrap(~variable, scales='free') + 
  theme_bw() + theme_tufte() + ggtitle(paste(test, old_release, 'vs', new_release, 'non time series results'))
res[variable %in% c('RMSE_V', 'RMSE_H') & sample_round == 64, summary(diff)]
res[variable %in% c('RMSE_H') & sample_round == 80, summary(diff)]

plot_vars = c('Max_RAM_GB', 'Total_Time_P1_Hours', 'RMSLE_V', 'RMSLE_H')
ggplot(res[sample_round == 64 & variable %in% plot_vars & !is.na(diff),], aes_string(x=old_release, y=new_release)) + 
  geom_point() + geom_abline(slope=1, intercept=0) + 
  facet_wrap(~variable, scales='free') + 
  theme_bw() + theme_tufte() + ggtitle(paste(test, old_release, 'vs', new_release, 'non time series results'))
res[variable %in% c('RMSLE_V', 'RMSLE_H') & sample_round == 64, summary(diff)]
res[variable %in% c('RMSLE_H') & sample_round == 80, summary(diff)]

######################################################
# Timing
######################################################
res[variable %in% c('Total_Time_P1_Hours') & !is.na(diff), list(
  min=round(min(diff)*60),
  mean=round(mean(diff)*60),
  median=round(median(diff)*60),
  max=round(max(diff)*60),
  .N
), by=list(sample_round = round(sample_round))][sample_round %in% c(16, 32, 64, 80),]

######################################################
# Table of results - valid
######################################################

vars = c('Filename', 'variable', old_release, new_release, 'diff')
res_normal = res[variable == 'Gini_V' & abs(diff) > 0 & sample_round==64, vars, with=F]
res_ts = res[variable == 'MASE_V' & diff > 0, vars, with=F]

values = c(old_release, new_release, 'diff')
res_normal = dcast.data.table(res_normal, Filename ~ variable, value.var = values)
res_ts = dcast.data.table(res_ts, Filename ~ variable, value.var = values)

res_cat <- copy(dat)
res_cat <- res_cat[!is.na(Max_RAM_GB),]
res_cat <- res_cat[!is.na(Total_Time_P1_Hours),]
res_cat <- res_cat[!is.na(`Gini Norm_P1`),]

res_cat <- res_cat[,list(
  best_gini_model = main_task[which.max(`Gini Norm_P1`)],
  best_mase_model = main_task[which.min(MASE_P1)]
), by=c('run', 'Filename', 'sample_round')]

measures = c('best_gini_model', 'best_mase_model')
res_cat = melt.data.table(res_cat, measure.vars=intersect(names(res_cat), measures))
res_cat = dcast.data.table(res_cat, Filename + sample_round + variable ~ run, value.var='value')

cat_norm = res_cat[sample_round==64 & variable == 'best_gini_model',]
cat_ts = res_cat[sample_round==0 & variable == 'best_mase_model',]

values = c(old_release, new_release)
cat_norm = dcast.data.table(cat_norm, Filename ~ variable, value.var = values)
cat_ts = dcast.data.table(cat_ts, Filename ~ variable, value.var = values)

res_normal = merge(res_normal, cat_norm, by='Filename')[order(diff_Gini_V),]
res_ts = merge(res_ts, cat_ts, by='Filename')[order(diff_MASE_V),]

res_normal
res_ts

fwrite(res_normal, '~/Downloads/non_ts_valid.txt')
fwrite(res_ts, '~/Downloads/ts_valid.txt')

######################################################
# Table of results - holdout
######################################################

vars = c('Filename', 'variable', old_release, new_release, 'diff')
res_normal = res[variable == 'Gini_H' & abs(diff) > 0 & sample_round==64, vars, with=F]
res_ts = res[variable == 'MASE_H' & diff > 0, vars, with=F]

values = c(old_release, new_release, 'diff')
res_normal = dcast.data.table(res_normal, Filename ~ variable, value.var = values)
res_ts = dcast.data.table(res_ts, Filename ~ variable, value.var = values)

res_cat <- copy(dat)
res_cat <- res_cat[!is.na(Max_RAM_GB),]
res_cat <- res_cat[!is.na(Total_Time_P1_Hours),]
res_cat <- res_cat[!is.na(`Gini Norm_H`),]

res_cat <- res_cat[,list(
  best_gini_model = main_task[which.max(`Gini Norm_H`)],
  best_mase_model = main_task[which.min(MASE_H)]
), by=c('run', 'Filename', 'sample_round')]

measures = c('best_gini_model', 'best_mase_model')
res_cat = melt.data.table(res_cat, measure.vars=intersect(names(res_cat), measures))
res_cat = dcast.data.table(res_cat, Filename + sample_round + variable ~ run, value.var='value')

cat_norm = res_cat[sample_round==64 & variable == 'best_gini_model',]
cat_ts = res_cat[sample_round==0 & variable == 'best_mase_model',]

values = c(old_release, new_release)
cat_norm = dcast.data.table(cat_norm, Filename ~ variable, value.var = values)
cat_ts = dcast.data.table(cat_ts, Filename ~ variable, value.var = values)

res_normal = merge(res_normal, cat_norm, by='Filename')[order(diff_Gini_H),]
res_ts = merge(res_ts, cat_ts, by='Filename')[order(diff_MASE_H),]

res_normal
res_ts

fwrite(res_normal, '~/Downloads/non_ts_holdout.txt')
fwrite(res_ts, '~/Downloads/ts_holdout.txt')

######################################################
# Lookit issues
######################################################

res[
  sample_round == 64 & variable %in% 'Total_Time_P1_Hours' & !is.na(diff),][which.max(diff),]

res[
  sample_round == 0 & variable %in% 'Total_Time_P1_Hours' & !is.na(diff),][which.min(diff),]

res[sample_round == 64 & variable %in% plot_vars & !is.na(diff) & diff > 1,]
