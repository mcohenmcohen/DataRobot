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
  '5b14203287734e0001c23194',
  '5b187956e7c1ab0001742ea1',
  '5b1e7f6dd46af50001ffec6a',
  '5b22fb59218f640001fa6219',
  '5b271ab82d1eef000180c99e',
  '5b27ae269c37ec0001a91792',
  '5b14205d87734e0001e8ff9d', # large from here
  '5b1dbc1c04d1e100016c9b40',
  '5b2298ff47bf3b0001378191',
  '5b22992d47bf3b000141fead',
  '5b271c0363ae910001fc4a55',
  '5b92cf91bcf8b20001bf7d92'
  )
new_mbtest_ids <- c(
  '5b7d7393e71d470001341dd0',
  '5b8027f1d32a5b00019bec6b',
  '5b92cf08fd8dc60001a347cd',
  '5b7daad920d52a00018df3d6', # large from here
  '5b80283ed32a5b00013b5282',
  '5b83f5ea13301c000164d8c7',
  '5b8ddad244ca3000019c1292',
  '5b9122a1cd432a0001d714ed', # scaleout
  '5b92cf24fd8dc6000115c4fa'
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

dat_old[,run := '4.3']
dat_new[,run := '4.4']

stopifnot(dat_old[,all(Metablueprint=='Metablueprint v12.0.01-so' | Metablueprint=='')])
stopifnot(dat_new[,all(Metablueprint=='Metablueprint v12.0.02-so')])

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
# Summarize stats
######################################################

res <- copy(dat)
res <- res[!is.na(Max_RAM_GB),]
res <- res[!is.na(Total_Time_P1_Hours),]
res <- res[!is.na(`Gini Norm_H`),]

res <- res[,list(
  Max_RAM_GB = max(Max_RAM_GB),
  Total_Time_P1_Hours = max(Total_Time_P1_Hours),
  Gini_V = max(`Gini Norm_P1`),
  Gini_H = max(`Gini Norm_H`),
  Gini_P = max(`Prediction Gini Norm`),
  MASE_H = min(`MASE_H`),
  MASE_V = min(`MASE_P1`)
), by=c('run', 'Filename', 'sample_round')]

measures = c(
  'Max_RAM_GB', 'Total_Time_P1_Hours', 'Gini_V', 'Gini_H', 'Gini_P', 'MASE_H', 'MASE_V')
res = melt.data.table(res, measure.vars=intersect(names(res), measures))
res = dcast.data.table(res, Filename + sample_round + variable ~ run, value.var='value')

res[,diff := as.numeric(`4.4`) - as.numeric(`4.3`)]
# res[abs(diff) < 0.01 & variable == 'Gini_H' & (sample_round == 0 | sample_round == 64),]

######################################################
# Plot of results
######################################################

plot_vars = c('Max_RAM_GB', 'Total_Time_P1_Hours', 'Gini_V', 'Gini_H')
ggplot(res[sample_round == 64 & variable %in% plot_vars,], aes(x=`4.3`, y=`4.4`)) + 
  geom_point() + geom_abline(slope=1, intercept=0) + 
  facet_wrap(~variable, scales='free') + 
  theme_bw() + theme_tufte() + ggtitle('4.4 vs 4.3 non time series results (YARN)')

plot_vars = c('Max_RAM_GB', 'Total_Time_P1_Hours', 'MASE_V', 'MASE_H')
ggplot(res[sample_round == 0 & variable %in% plot_vars,], aes(x=`4.3`, y=`4.4`)) + 
  geom_point() + geom_abline(slope=1, intercept=0) + 
  facet_wrap(~variable, scales='free') + 
  theme_bw() + theme_tufte() + ggtitle('4.4 vs 4.3 time series results (YARN)')

######################################################
# Table of results
######################################################

res_normal = res[variable == 'Gini_H' & abs(diff) > 0.01 & sample_round==64,
                 list(Filename, variable, `4.3`, `4.4`, diff)]
res_ts = res[variable == 'MASE_H' & diff > 0,
             list(Filename, variable, `4.3`, `4.4`, diff)]

values = c('4.3', '4.4', 'diff')
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

values = c('4.3', '4.4')
cat_norm = dcast.data.table(cat_norm, Filename ~ variable, value.var = values)
cat_ts = dcast.data.table(cat_ts, Filename ~ variable, value.var = values)

res_normal = merge(res_normal, cat_norm, by='Filename')[order(diff_Gini_H),]
res_ts = merge(res_ts, cat_ts, by='Filename')[order(diff_MASE_H),]

res_normal
res_ts
