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

old_release <- 'py34'
new_release <- 'py37'
test <- 'DSS_37'

new_mbtest_ids <- c(
  '5df40ad540ae7c000cdfba3d'
)

old_mbtest_ids <- c(
  '5df40b0f40ae7c000b761692'
)

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
# Summarize stats - numeric 
######################################################

res <- copy(dat)
res <- res[!is.na(Max_RAM_GB),]
res <- res[!is.na(Total_Time_P1_Hours),]
res <- res[!is.na(`Gini Norm_H`),]

res_num <- res[,list(
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
  tmp = sort(unique(res_num[[v]]))
  wont_convert = !is.finite(as.numeric(tmp))
  if(any(wont_convert)){
    print(tmp[wont_convert])
  }
  set(res_num, j=v, value=as.numeric(res_num[[v]]))
}
res_num = melt.data.table(res_num, measure.vars=intersect(names(res_num), measures))
res_num = dcast.data.table(res_num, Filename + Y_Type + sample_round + variable ~ run, value.var='value')

set(res_num, j='diff', value = res_num[[new_release]] - res_num[[old_release]])

######################################################
# Summarize stats - cat 
######################################################

res_cat <- res[,list(
  Gini_H = main_task[which.max(`Gini Norm_H`)],
  Logloss_H = main_task[which.min(`LogLoss_H`)]
), by=c('run', 'Filename', 'Y_Type', 'sample_round')]
res_cat[,run := paste0(run, '_best_model')]
res_cat = melt.data.table(res_cat, measure.vars=intersect(names(res_cat), measures))
res_cat = dcast.data.table(res_cat, Filename + Y_Type + sample_round + variable ~ run, value.var='value')

res = merge(res_cat, res_num, by=intersect(names(res_cat), names(res_num)), all=TRUE)

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
# Lookit issues
######################################################

res[,pct_diff := (get(new_release) / get(old_release)) - 1]
res[abs(pct_diff) > 0.05,][order(variable, diff),]

# Accuracy changes
res[variable=='Gini_V' & sample_round==64 & abs(diff) > 0.01,][order(variable),]
res[variable=='Gini_H' & sample_round==64 & abs(diff) > 0.01,][order(variable),]
res[variable=='Logloss_H' & sample_round==64 & abs(diff) > 0.01,][order(variable),]
res[variable=='Logloss_V' & sample_round==64 & abs(diff) > 0.01,][order(variable),]
