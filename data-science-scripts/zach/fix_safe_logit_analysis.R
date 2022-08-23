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
  #'5c3cec097347c9002b0f404f'  # Something wrong with this test
  '5c3f7e527347c90029592c34',  # Dev MBP
  '5c4090667347c90024f8436e',  # Full MBP
  '5c41fbaf7347c9002bf34388'  # Full MBP on master
)
new_mbtest_ids <- c(
  #'5c3e3a497347c900241262ff'  # Something wrong with this test
  '5c3f7e4b7347c900241263ac',  # Dev MBP
  '5c4090887347c90028591515',  # Full MBP
  '5c45d0ec7347c90026d97c9d'  # Full MBP on master
)

old_release <- 'bad_logit'
new_release <- 'fixed_logit'
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

#dat <- dat[is_prime == FALSE,]

######################################################
# Summarize plot + stats - with main task
######################################################

res <- copy(dat)
res <- res[!is.na(Max_RAM_GB),]
res <- res[!is.na(Total_Time_P1_Hours),]
res <- res[!is.na(`Gini Norm_H`),]

res <- res[,list(
  Max_RAM_GB = max(Max_RAM_GB),
  Total_Time_P1_Hours = max(Total_Time_P1_Hours),
  LogLoss_P1 = min(LogLoss_P1),
  LogLoss_H = min(LogLoss_H)
), by=c('run', 'Filename', 'sample_round', 'main_task')]

measures = c(
  'Max_RAM_GB', 'Total_Time_P1_Hours', 'LogLoss_P1', 'LogLoss_H')
for(v in measures){
  tmp = sort(unique(res[[v]]))
  wont_convert = !is.finite(as.numeric(tmp))
  if(any(wont_convert)){
    print(tmp[wont_convert])
  }
  set(res, j=v, value=as.numeric(res[[v]]))
}
res = melt.data.table(res, measure.vars=intersect(names(res), measures))
res = dcast.data.table(res, Filename + sample_round + variable + main_task ~ run, value.var='value')

set(res, j='diff', value = res[[new_release]] - res[[old_release]])
# res[abs(diff) < 0.01 & variable == 'Gini_H' & (sample_round == 0 | sample_round == 64),]

# Plot of results
plot_vars = c('Max_RAM_GB', 'Total_Time_P1_Hours', 'LogLoss_P1', 'LogLoss_H')
ggplot(res[sample_round == 64 & variable %in% plot_vars & !is.na(diff),], aes_string(x=old_release, y=new_release)) + 
  geom_point() + geom_abline(slope=1, intercept=0) + 
  facet_wrap(~variable, scales='free') + 
  theme_bw() + theme_tufte() + ggtitle(paste(test, old_release, 'vs', new_release, 'non time series results'))

# Table of results - Holdout
vars = c('Filename', 'variable', 'main_task', old_release, new_release, 'diff')
values = c(old_release, new_release, 'diff')
res_normal = res[variable == 'LogLoss_H' & sample_round==64, vars, with=F]
res_normal = dcast.data.table(res_normal, Filename + main_task ~ variable, value.var = values)
res_normal = res_normal[order(diff_LogLoss_H),][diff_LogLoss_H!=0,]
res_normal[,hist(diff_LogLoss_H)]
res_normal[,summary(diff_LogLoss_H)]
res_normal[,summary(diff_LogLoss_H/bad_logit_LogLoss_H)]
res_normal

# Table of results - Validation
vars = c('Filename', 'variable', 'main_task', old_release, new_release, 'diff')
values = c(old_release, new_release, 'diff')
res_normal = res[variable == 'LogLoss_P1' & sample_round==64, vars, with=F]
res_normal = dcast.data.table(res_normal, Filename + main_task ~ variable, value.var = values)
res_normal = res_normal[order(diff_LogLoss_P1),][diff_LogLoss_P1!=0,]
res_normal[,hist(diff_LogLoss_P1)]
res_normal[,summary(diff_LogLoss_P1)]
res_normal[,summary(diff_LogLoss_P1/bad_logit_LogLoss_P1)]
res_normal

######################################################
# Summarize plot + stats - overall at the project level, no main task
######################################################

res <- copy(dat)
res <- res[!is.na(Max_RAM_GB),]
res <- res[!is.na(Total_Time_P1_Hours),]
res <- res[!is.na(`Gini Norm_H`),]

res <- res[,list(
  Max_RAM_GB = max(Max_RAM_GB),
  Total_Time_P1_Hours = max(Total_Time_P1_Hours),
  LogLoss_P1 = min(LogLoss_P1),
  LogLoss_H = min(LogLoss_H)
), by=c('run', 'Filename', 'sample_round')]

measures = c(
  'Max_RAM_GB', 'Total_Time_P1_Hours', 'LogLoss_P1', 'LogLoss_H')
for(v in measures){
  tmp = sort(unique(res[[v]]))
  wont_convert = !is.finite(as.numeric(tmp))
  if(any(wont_convert)){
    print(tmp[wont_convert])
  }
  set(res, j=v, value=as.numeric(res[[v]]))
}
res = melt.data.table(res, measure.vars=intersect(names(res), measures))
res = dcast.data.table(res, Filename + sample_round + variable ~ run, value.var='value')

set(res, j='diff', value = res[[new_release]] - res[[old_release]])
# res[abs(diff) < 0.01 & variable == 'Gini_H' & (sample_round == 0 | sample_round == 64),]

# Plot of results
plot_vars = c('Max_RAM_GB', 'Total_Time_P1_Hours', 'LogLoss_P1', 'LogLoss_H')
ggplot(res[sample_round == 64 & variable %in% plot_vars & !is.na(diff),], aes_string(x=old_release, y=new_release)) + 
  geom_point() + geom_abline(slope=1, intercept=0) + 
  facet_wrap(~variable, scales='free') + 
  theme_bw() + theme_tufte() + ggtitle(paste(test, old_release, 'vs', new_release, 'non time series results'))

# Table of results - Holdout
vars = c('Filename', 'variable', old_release, new_release, 'diff')
values = c(old_release, new_release, 'diff')
res_normal = res[variable == 'LogLoss_H' & sample_round==64, vars, with=F]
res_normal = dcast.data.table(res_normal, Filename ~ variable, value.var = values)
res_normal = res_normal[order(diff_LogLoss_H),][diff_LogLoss_H!=0,]
res_normal[,hist(diff_LogLoss_H)]
res_normal[,summary(diff_LogLoss_H)]
res_normal[,summary(diff_LogLoss_H/bad_logit_LogLoss_H)]
res_normal

# Table of results - Validation
vars = c('Filename', 'variable', old_release, new_release, 'diff')
values = c(old_release, new_release, 'diff')
res_normal = res[variable == 'LogLoss_P1' & sample_round==64, vars, with=F]
res_normal = dcast.data.table(res_normal, Filename ~ variable, value.var = values)
res_normal = res_normal[order(diff_LogLoss_P1),][diff_LogLoss_P1!=0,]
res_normal[,hist(diff_LogLoss_P1)]
res_normal[,summary(diff_LogLoss_P1)]
res_normal[,summary(diff_LogLoss_P1/bad_logit_LogLoss_P1)]
res_normal