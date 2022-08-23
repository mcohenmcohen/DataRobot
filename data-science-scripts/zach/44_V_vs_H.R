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

new_mbtest_ids <- c(
  '5b7cdbd3719cdb00016b3d1a',
  '5b893a7e9c3f2c0001ef72d8',
  '5b9122863be021000194f178'
)

prefix = 'http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests='
suffix = '&max_sample_size_only=false'
new_mbtest_urls <- paste0(prefix, new_mbtest_ids, suffix)
dat_new_raw <- pblapply(new_mbtest_urls, fread)

######################################################
# Convert possible int64s to numeric
######################################################

dat_new <- copy(dat_new_raw)

clean_data <- function(x){
  x[,Max_RAM_GB := as.numeric(Max_RAM / 1e9)]
  x[,Total_Time_P1_Hours := as.numeric(Total_Time_P1 / 3600)]
  x[,size_GB := as.numeric(size / 1e9)]
  x[,dataset_size_GB := as.numeric(dataset_size / 1e9)]
  return(x)
}

dat_new <- lapply(dat_new, clean_data)

stopifnot(all(sapply(dat_new, function(x) 'Max_RAM_GB' %in% names(x))))

######################################################
# Combine data
######################################################
get_names <- function(x){
  not_int64 <- sapply(x,  class) != 'integer64'
  names(x)[not_int64]
}

names_all <- Reduce(intersect, lapply(dat_new, get_names))
stopifnot('Metablueprint' %in% names_all)

dat_new <- lapply(dat_new, function(x) x[,names_all,with=F])
dat_new <- rbindlist(dat_new, use.names=T)
dat_new <- dat_new[!grepl("Z", dat_new$training_length),]
dat_new[,run := '4.4']
stopifnot(dat_new[,all(Metablueprint=='Metablueprint v12.0.02-so')])

dat <- copy(dat_new)

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

# dat <- dat[is_prime == FALSE,]

######################################################
# Summarize stats
######################################################

# Subset
res <- copy(dat)
res <- res[sample_round != 0,]
res <- res[sample_round <= 64,]
res = res[!is.na(LogLoss_H),]

# Select
res <- res[,list(
  Filename,
  Max_RAM_GB = Max_RAM_GB,
  Total_Time_P1_Hours = Total_Time_P1_Hours,
  LogLoss_H = LogLoss_H,
  LogLoss_V = LogLoss_P1,
  LogLoss_CV = `LogLoss_P1-5`,
  LogLoss_P = `Prediction LogLoss`,
  AUC_H = AUC_H,
  AUC_V = AUC_P1,
  AUC_CV = `AUC_P1-5`,
  AUC_P = `Prediction AUC`)]

######################################################
# Plot of results
######################################################

# Logloss
ggplot(res[!is.na(LogLoss_H),], aes(x=LogLoss_H, y=LogLoss_V)) + 
  geom_point() + geom_smooth(method='lm') + 
  geom_abline(slope=1, intercept=0) + 
  theme_bw() + theme_tufte() + ggtitle('Validation vs Holdout LogLoss')

ggplot(res[!is.na(LogLoss_CV),], aes(x=LogLoss_H, y=LogLoss_CV)) + 
  geom_point() + geom_smooth(method='lm') + 
  geom_abline(slope=1, intercept=0) + 
  theme_bw() + theme_tufte() + ggtitle('Cross-Validation vs Holdout LogLoss')

# AUC
res[!is.na(AUC_V),.N]
res[!is.na(AUC_V),length(unique(Filename))]
ggplot(res[!is.na(AUC_V),], aes(x=AUC_H, y=AUC_V)) + 
  geom_point() + geom_smooth(method='lm') + 
  geom_abline(slope=1, intercept=0) + 
  theme_bw() + theme_tufte() + ggtitle('Validation vs Holdout AUC')

res[!is.na(AUC_CV),.N]
res[!is.na(AUC_CV),length(unique(Filename))]
ggplot(res[!is.na(AUC_CV),], aes(x=AUC_H, y=AUC_CV)) + 
  geom_point() + geom_smooth(method='lm') + 
  geom_abline(slope=1, intercept=0) + 
  theme_bw() + theme_tufte() + ggtitle('Cross-Validation vs Holdout AUC')

######################################################
# Table of results
######################################################
summary(res[!is.na(LogLoss_V),][,list(LogLoss_V, LogLoss_CV, LogLoss_H)])
summary(res[!is.na(LogLoss_CV),][,list(LogLoss_V, LogLoss_CV, LogLoss_H)])

summary(res[!is.na(AUC_V),][,list(AUC_V, AUC_CV, AUC_H)])
summary(res[!is.na(AUC_CV),][,list(AUC_V, AUC_CV, AUC_H)])
