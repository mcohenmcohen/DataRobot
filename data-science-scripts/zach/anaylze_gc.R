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
  '5d56b8d97347c9002462e87e'
)
new_mbtest_ids <- c(
  '5d5c041d7347c900283d82d7'
)

old_release <- 'master'
new_release <- 'with_gc'
test <- 'Test of adding gc()'

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

res <- res[,list(
  Max_RAM_GB = max(Max_RAM_GB),
  Total_Time_P1_Hours = max(Total_Time_P1_Hours),
  Gini_V = max(`Gini Norm_P1`),
  Gini_H = max(`Gini Norm_H`),
  Gini_P = max(`Prediction Gini Norm`),
  Logloss_V = min(`LogLoss_P1`),
  Logloss_H = min(`LogLoss_H`),
  MASE_H = min(`MASE_H`),
  MASE_V = min(`MASE_P1`)
), by=c('run', 'Filename', 'Y_Type', 'sample_round')]

measures = c(
  'Max_RAM_GB', 'Total_Time_P1_Hours', 'Gini_V', 'Gini_H', 'Gini_P', 'Logloss_V', 'Logloss_H', 'MASE_H', 'MASE_V')
for(v in measures){
  tmp = sort(unique(res[[v]]))
  wont_convert = !is.finite(as.numeric(tmp))
  if(any(wont_convert)){
    print(tmp[wont_convert])
  }
  set(res, j=v, value=as.numeric(res[[v]]))
}
res = melt.data.table(res, measure.vars=intersect(names(res), measures))
res = dcast.data.table(res, Filename + Y_Type + sample_round + variable ~ run, value.var='value')

set(res, j='diff', value = res[[new_release]] - res[[old_release]])
# res[abs(diff) < 0.01 & variable == 'Gini_H' & (sample_round == 0 | sample_round == 64),]

######################################################
# Plot of results
######################################################

plot_vars = c('Max_RAM_GB', 'Total_Time_P1_Hours', 'Gini_V', 'Gini_H')
ggplot(res[sample_round == 64 & variable %in% plot_vars & !is.na(diff),], aes_string(x=old_release, y=new_release, color='Y_Type')) + 
  geom_point() + geom_abline(slope=1, intercept=0) + 
  facet_wrap(~variable, scales='free') + 
  theme_bw() + theme_tufte() + ggtitle(paste(test, old_release, 'vs', new_release, 'non time series results'))

plot_vars = c('Max_RAM_GB', 'Total_Time_P1_Hours', 'MASE_V', 'MASE_H')
ggplot(res[sample_round == 0 & variable %in% plot_vars & !is.na(diff),], aes_string(x=old_release, y=new_release, color='Y_Type')) + 
  geom_point() + geom_abline(slope=1, intercept=0) + 
  facet_wrap(~variable, scales='free') + 
  theme_bw() + theme_tufte() + ggtitle(paste(test, old_release, 'vs', new_release, 'time series results'))

plot_vars = c('Max_RAM_GB', 'Total_Time_P1_Hours', 'Logloss_V', 'Logloss_H')
ggplot(res[sample_round == 64 & variable %in% plot_vars & !is.na(diff) & Y_Type %in% c('Multiclass', 'Binary'),], aes_string(x=old_release, y=new_release, color='Y_Type')) + 
  geom_point() + geom_abline(slope=1, intercept=0) + 
  facet_wrap(~variable, scales='free') + 
  theme_bw() + theme_tufte() + ggtitle(paste(test, old_release, 'vs', new_release, 'non time series results - classification only'))

######################################################
# Table of results
######################################################

vars = c('Filename', 'Y_Type', 'variable', old_release, new_release, 'diff')
res_normal = res[variable == 'Gini_H' & abs(diff) > 0.01 & sample_round==64, vars, with=F]
res_ts = res[variable == 'MASE_H' & diff > 0, vars, with=F]

values = c(old_release, new_release, 'diff')
res_normal = dcast.data.table(res_normal, Filename + Y_Type ~ variable, value.var = values)
res_ts = dcast.data.table(res_ts, Filename + Y_Type ~ variable, value.var = values)

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

######################################################
# Lookit issues
######################################################

res[variable %in% c('Max_RAM_GB'),][order(diff),]

# Runtime/ram issues
res[,pct_diff := (get(new_release) / get(old_release)) - 1]

# Runtime diffs
res[sample_round==64 & diff > 0.1 & pct_diff > 0.25 & variable %in% c('Total_Time_P1_Hours'),][order(pct_diff),]

# Non TS
res[sample_round==64 & diff > 0.1 & pct_diff > 0.25 & variable %in% c('Total_Time_P1_Hours', 'Max_RAM_GB'),][order(pct_diff),]

# TS
res[sample_round==0 & diff > 0.1 & pct_diff > 0.25 & variable %in% c('Total_Time_P1_Hours', 'Max_RAM_GB'),][order(pct_diff),]

res[
  sample_round == 64 & variable %in% 'Total_Time_P1_Hours' & !is.na(diff),][which.max(diff),]

res[
  sample_round == 0 & variable %in% 'Total_Time_P1_Hours' & !is.na(diff),][which.min(diff),]

res[sample_round == 64 & variable %in% plot_vars & !is.na(diff) & diff > 1,]

# 100_multiclass_AirlineFlights2008-reduced_0.95_downsampled.csv 
dat[Filename=='100_multiclass_AirlineFlights2008-reduced_0.95_downsampled.csv' & run == old_release,][which.max(Total_Time_P1), main_task]
dat[Filename=='100_multiclass_AirlineFlights2008-reduced_0.95_downsampled.csv' & run == new_release,][which.max(Total_Time_P1), main_task]

# 100_multiclass_AirlineFlights2008-reduced_0.95_downsampled.csv 
dat[Filename=='100_multiclass_AirlineFlights2008-reduced_0.95_downsampled.csv' & run == old_release,][which.max(Total_Time_P1), Blueprint]
dat[Filename=='100_multiclass_AirlineFlights2008-reduced_0.95_downsampled.csv' & run == new_release,][which.max(Total_Time_P1), Blueprint]

# 100_multiclass_AirlineFlights2008-reduced_0.95_downsampled.csv 
dat[Filename=='100_multiclass_AirlineFlights2008-reduced_0.95_downsampled.csv' & run == old_release,][which.max(Total_Time_P1), `_id`]
dat[Filename=='100_multiclass_AirlineFlights2008-reduced_0.95_downsampled.csv' & run == new_release,][which.max(Total_Time_P1), `_id`]



# bloggers_small_80.csv
dat[Filename=='bloggers_small_80.csv' & run == old_release,][which.max(Max_RAM_GB), main_task]
dat[Filename=='bloggers_small_80.csv' & run == new_release,][which.max(Max_RAM_GB), main_task]

dat[Filename=='bloggers_small_80.csv' & run == old_release, max(`Gini Norm_P1`)]
dat[Filename=='bloggers_small_80.csv' & run == new_release, max(`Gini Norm_P1`)]

dat[Filename=='bloggers_small_80.csv' & run == old_release, max(`Gini Norm_H`)]
dat[Filename=='bloggers_small_80.csv' & run == new_release, max(`Gini Norm_H`)]

# yelp_review_polarity_full_0.95.csv
dat[Filename=='yelp_review_polarity_full_0.95.csv' & run == old_release,][which.max(Max_RAM_GB), main_task]
dat[Filename=='yelp_review_polarity_full_0.95.csv' & run == new_release,][which.max(Max_RAM_GB), main_task]

dat[Filename=='yelp_review_polarity_full_0.95.csv' & run == old_release, max(`Gini Norm_P1`)]
dat[Filename=='yelp_review_polarity_full_0.95.csv' & run == new_release, max(`Gini Norm_P1`)]

dat[Filename=='yelp_review_polarity_full_0.95.csv' & run == old_release, max(`Gini Norm_H`)]
dat[Filename=='yelp_review_polarity_full_0.95.csv' & run == new_release, max(`Gini Norm_H`)]
