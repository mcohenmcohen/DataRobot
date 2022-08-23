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

old_mbtest_ids <- c(
  '5ce59a7ade57df00018888c7',
  '5cf176166af6610001182b71',
  '5cf69eba572f570001697a9a',
  '5cfe6941b7815d0001f94cf7',
  '5ce7e7516f435d0001a735e5',
  '5cfab59fad93bf00010041f4'
  )
new_mbtest_ids <- c(
  '5d83795da28ae300010c8b98',
  '5d8183a6116a320001d04057',
  '5d7c33d2eaca2200014f77d8',
  '5d852e2072d75500014f765b',
  '5d7fd7aaa21cf600018cf169',
  '5d7c0b1e47a776000180950b'
)

old_release <- 'v5.1'
new_release <- 'v5.2'
test <- 'Docker'

prefix = 'http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests='
suffix = '&max_sample_size_only=false'

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
), by=c('run', 'Filename', 'Y_Type', 'sample_round')]  # , 'main_task'

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
res = dcast.data.table(res, Filename + Y_Type + sample_round + variable ~ run, value.var='value')  # + main_task

set(res, j='diff', value = res[[new_release]] - res[[old_release]])
# res[abs(diff) < 0.01 & variable == 'Gini_H' & (sample_round == 0 | sample_round == 64),]

######################################################
# Plot of results
######################################################

# plot_vars = c('Max_RAM_GB', 'Total_Time_P1_Hours', 'Logloss_V', 'Gini_V')
# ggplot(res[sample_round == 64 & variable %in% plot_vars & !is.na(diff),], aes_string(x=old_release, y=new_release, color='Y_Type')) + 
#   geom_point() + geom_abline(slope=1, intercept=0) + 
#   facet_wrap(~variable, scales='free') + 
#   theme_bw() + theme_tufte() + ggtitle(paste(test, old_release, 'vs', new_release, 'non time series results'))

plot_vars = c('Max_RAM_GB', 'Total_Time_P1_Hours', 'Logloss_H', 'Gini_H')
ggplot(res[sample_round == 64 & variable %in% plot_vars & !is.na(diff),], aes_string(x=old_release, y=new_release, color='Y_Type')) + 
  geom_point() + geom_abline(slope=1, intercept=0) + 
  facet_wrap(~variable, scales='free') + 
  theme_bw() + theme_tufte() + ggtitle(paste(test, old_release, 'vs', new_release, 'non time series results'))

# plot_vars = c('Max_RAM_GB', 'Total_Time_P1_Hours', 'Logloss_V', 'MASE_V')
# ggplot(res[sample_round == 0 & variable %in% plot_vars & !is.na(diff),], aes_string(x=old_release, y=new_release, color='Y_Type')) + 
#   geom_point() + geom_abline(slope=1, intercept=0) + 
#   facet_wrap(~variable, scales='free') + 
#   theme_bw() + theme_tufte() + ggtitle(paste(test, old_release, 'vs', new_release, 'time series results'))

plot_vars = c('Max_RAM_GB', 'Total_Time_P1_Hours', 'Logloss_H', 'MASE_H')
ggplot(res[sample_round == 0 & variable %in% plot_vars & !is.na(diff),], aes_string(x=old_release, y=new_release, color='Y_Type')) +
  geom_point() + geom_abline(slope=1, intercept=0) +
  facet_wrap(~variable, scales='free') +
  theme_bw() + theme_tufte() + ggtitle(paste(test, old_release, 'vs', new_release, 'time series results'))

######################################################
# Table of results
######################################################

vars = c('Filename', 'Y_Type', 'variable', old_release, new_release, 'diff')
res_normal = res[variable == 'Logloss_H' & abs(diff) > 0.01 & sample_round==64, vars, with=F]
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

# Runtime/ram issues
res[,pct_diff := (get(new_release) / get(old_release)) - 1]

# diffs - non TS
ss = 64
res[sample_round==ss & diff > 0.1 & pct_diff > 0.25 & variable %in% c('Total_Time_P1_Hours'),][order(diff),list(Filename, Y_Type, `v5.1`, `v5.2`, diff)]
res[sample_round==ss & diff > .5 & pct_diff > 0.25 & variable %in% c('Max_RAM_GB'),][order(diff),list(Filename, Y_Type, `v5.1`, `v5.2`, diff)]
res[sample_round==ss & diff > 0.01 & pct_diff > 0.25 & variable %in% c('Gini_H'),][order(diff),list(Filename, Y_Type, `v5.1`, `v5.2`, diff)]
res[sample_round==ss & diff > 0.01 & pct_diff > 0.1 & variable %in% c('Logloss_H'),][order(diff),list(Filename, Y_Type, `v5.1`, `v5.2`, diff)]

# diffs - TS
ss = 0
res[sample_round==ss & diff > 0.1 & pct_diff > 0.25 & variable %in% c('Total_Time_P1_Hours'),][order(diff),list(Filename, Y_Type, `v5.1`, `v5.2`, diff)]
res[sample_round==ss & diff > .5 & pct_diff > 0.25 & variable %in% c('Max_RAM_GB'),][order(diff),list(Filename, Y_Type, `v5.1`, `v5.2`, diff)]
res[sample_round==ss & diff > 0.01 & pct_diff > 0.25 & variable %in% c('Gini_H'),][order(diff),list(Filename, Y_Type, `v5.1`, `v5.2`, diff)]
res[sample_round==ss & diff > 0.01 & pct_diff > 0.1 & variable %in% c('Logloss_H'),][order(diff),list(Filename, Y_Type, `v5.1`, `v5.2`, diff)]

######################################################
# Lookit Datasets
######################################################

# phone_gender_narrow_leak.csv
x <- 'phone_gender_narrow_leak.csv'
dat[Filename==x & run == old_release,][which.max(Total_Time_P1), list(main_task, mbtestid, AUC_P1, Total_Time_P1)]
dat[Filename==x & run == new_release,][which.max(Total_Time_P1), list(main_task, mbtestid, AUC_P1, Total_Time_P1)]

res[Filename==x & variable=='Total_Time_P1_Hours',list(
  old=max(get(old_release), na.rm=T),
  new=max(get(new_release), na.rm=T),
  main_task_old=main_task[which.max(get(old_release))],
  main_task_new=main_task[which.max(get(new_release))]
), by=sample_round]

res[Filename==x & variable=='Logloss_H',list(
  old=min(get(old_release), na.rm=T),
  new=min(get(new_release), na.rm=T),
  main_task_old=main_task[which.min(get(old_release))],
  main_task_new=main_task[which.min(get(new_release))]
), by=sample_round]

dat[Filename==x & run == old_release,][which.max(Max_RAM_GB), main_task]
dat[Filename==x & run == new_release,][which.max(Max_RAM_GB), main_task]
dat[Filename==x & run == old_release,][which.max(Total_Time_P1_Hours), main_task]
dat[Filename==x & run == new_release,][which.max(Total_Time_P1_Hours), main_task]

# el-nino-friendly-30000_80.csv
x <- 'French_TP_claim_frequency_80.csv'
dat[Filename==x & run == old_release,][which.max(Total_Time_P1), list(main_task, mbtestid, AUC_P1, Total_Time_P1)]
dat[Filename==x & run == new_release,][which.max(Total_Time_P1), list(main_task, mbtestid, AUC_P1, Total_Time_P1)]

res[Filename==x & variable=='Max_RAM_GB',list(
  old=max(get(old_release), na.rm=T),
  new=max(get(new_release), na.rm=T),
  main_task_old=main_task[which.max(get(old_release))],
  main_task_new=main_task[which.max(get(new_release))]
), by=sample_round]

res[Filename==x & variable=='Logloss_H',list(
  old=min(get(old_release), na.rm=T),
  new=min(get(new_release), na.rm=T),
  main_task_old=main_task[which.min(get(old_release))],
  main_task_new=main_task[which.min(get(new_release))]
), by=sample_round]

dat[Filename==x & run == old_release,][which.max(Max_RAM_GB), main_task]
dat[Filename==x & run == new_release,][which.max(Max_RAM_GB), main_task]
dat[Filename==x & run == old_release,][which.max(Total_Time_P1_Hours), main_task]
dat[Filename==x & run == new_release,][which.max(Total_Time_P1_Hours), main_task]

x <- 'OMS_Train2.csv'
dat[Filename==x & run == old_release,][which.max(Max_RAM_GB), main_task]
dat[Filename==x & run == new_release,][which.max(Max_RAM_GB), main_task]
dat[Filename==x & run == old_release,][which.max(Total_Time_P1_Hours), main_task]
dat[Filename==x & run == new_release,][which.max(Total_Time_P1_Hours), main_task]

x <- 'cvs_all_drugs.csv'
dat[Filename==x & run == old_release,][which.max(Max_RAM_GB), main_task]
dat[Filename==x & run == new_release,][which.max(Max_RAM_GB), main_task]
dat[Filename==x & run == old_release,][which.max(Total_Time_P1_Hours), main_task]
dat[Filename==x & run == new_release,][which.max(Total_Time_P1_Hours), main_task]


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
dat[Filename=='yelp_review_polarity_full_0.95.csv' & run == old_release,][which.max(Total_Time_P1_Hours), main_task]
dat[Filename=='yelp_review_polarity_full_0.95.csv' & run == old_release,][which.max(Total_Time_P1_Hours), main_task]

dat[Filename=='yelp_review_polarity_full_0.95.csv' & run == old_release,][which.max(Total_Time_P1_Hours), Blueprint]
dat[Filename=='yelp_review_polarity_full_0.95.csv' & run == new_release,][which.max(Total_Time_P1_Hours), Blueprint]

dat[Filename=='yelp_review_polarity_full_0.95.csv' & run == old_release,][which.max(Max_RAM_GB), main_task]
dat[Filename=='yelp_review_polarity_full_0.95.csv' & run == new_release,][which.max(Max_RAM_GB), main_task]

dat[Filename=='yelp_review_polarity_full_0.95.csv' & run == old_release, max(`Gini Norm_P1`)]
dat[Filename=='yelp_review_polarity_full_0.95.csv' & run == new_release, max(`Gini Norm_P1`)]

dat[Filename=='yelp_review_polarity_full_0.95.csv' & run == old_release, max(`Gini Norm_H`)]
dat[Filename=='yelp_review_polarity_full_0.95.csv' & run == new_release, max(`Gini Norm_H`)]
