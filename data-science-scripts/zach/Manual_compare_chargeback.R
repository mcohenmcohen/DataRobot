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
  '5769d97bd9a75306c3a6fddd'
  )
new_mbtest_ids <- c(
  '576cffe07a760c72a70a548a'
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

old_release <- 'June_22'
new_release <- 'June_24'
test <- 'Chargeback'

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

stopifnot(dat_old[,all(Metablueprint=='Metablueprint v10.0.08-so' | Metablueprint=='')])
stopifnot(dat_new[,all(Metablueprint=='Metablueprint v10.0.08-so')])

dat <- rbindlist(list(dat_old, dat_new), use.names=T)

######################################################
# Add some vars
######################################################

dat[,dataset_bin := cut(dataset_size_GB, unique(c(0, 1.5, 2.5, 5, ceiling(max(dataset_size_GB)))), ordered_result=T, include.lowest=T)]
dat[,sample_round := Sample_Pct]
dat[sample_round=='--', sample_round := '0']
dat[,sample_round := round(as.numeric(sample_round))]

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

res[,sample_round := factor(sample_round)]
plot_vars = c('Max_RAM_GB', 'Total_Time_P1_Hours', 'Logloss_V', 'Logloss_H')
ggplot(res[variable %in% plot_vars & !is.na(diff) & Y_Type %in% c('Multiclass', 'Binary'),], aes_string(x=old_release, y=new_release, color='sample_round')) + 
  geom_point() + geom_abline(slope=1, intercept=0) + 
  facet_wrap(~variable, scales='free') + 
  theme_bw() + theme_tufte() + ggtitle(paste(test, old_release, 'vs', new_release, 'non time series results - classification only'))

######################################################
# chargeback_clean_80.csv deep dive
######################################################

x <- 'chargeback_clean_80.csv'
dat[Filename==x & run == old_release,][which.max(Max_RAM_GB), main_task]
dat[Filename==x & run == new_release,][which.max(Max_RAM_GB), main_task]
dat[Filename==x & run == old_release,][which.max(Total_Time_P1_Hours), main_task]
dat[Filename==x & run == new_release,][which.max(Total_Time_P1_Hours), main_task]

tbl <- dat[Filename==x,][,list(
  Logloss_V = min(LogLoss_P1, na.rm=T),
  Logloss_H = min(LogLoss_H, na.rm=T)
), by=c('Filename', 'run', 'sample_round', 'main_task')]
tbl <- dcast.data.table(tbl, Filename + sample_round + main_task ~ run, value.var=c('Logloss_V', 'Logloss_H'))
tbl[grepl("BLENDER", main_task), main_task := 'Blender']
tbl[grepl("LR", main_task), main_task := 'Linear']
tbl[grepl("ENETCD", main_task), main_task := 'Linear']
tbl[!is.finite(Logloss_V_June_22), Logloss_V_June_22 := NA]
tbl[!is.finite(Logloss_V_June_24), Logloss_V_June_24 := NA]
tbl[!is.finite(Logloss_H_June_22), Logloss_H_June_22 := NA]
tbl[!is.finite(Logloss_H_June_24), Logloss_H_June_24 := NA]

lim = c(0, 0.08)
colors <- c(
  "#1f78b4", "#ff7f00", "#6a3d9a", "#33a02c", "#e31a1c", "#b15928",
  "#a6cee3", "#fdbf6f", "#cab2d6", "#b2df8a", "#fb9a99", "black",
  "grey1", "grey10"
)

# all 
ggplot(tbl, aes(
  x=Logloss_H_June_22, 
  y=Logloss_H_June_24, 
  color=main_task
)) + 
  geom_point() + 
  geom_abline(slope=1, intercept=0) + 
  theme_tufte() + 
  theme(legend.position='top') + 
  scale_color_manual(values=colors) 

# By sample size
ggplot(tbl, aes(
  x=Logloss_H_June_22, 
  y=Logloss_H_June_24, 
  color=main_task
  )) + 
  geom_point() + 
  geom_abline(slope=1, intercept=0) + 
  theme_tufte() + 
  theme(legend.position='top') + 
  scale_color_manual(values=colors) +
  facet_wrap(~sample_round)

tbl <- dat[Filename==x,][,list(
  Logloss_V = min(LogLoss_P1, na.rm=T),
  Logloss_H = min(LogLoss_H, na.rm=T), 
  main_task_V = main_task[which.min(LogLoss_P1)],
  main_task_H = main_task[which.min(LogLoss_H)],
  blueprint_V = Blueprint[which.min(LogLoss_P1)],
  blueprint_H = Blueprint[which.min(LogLoss_H)]
), by=c('Filename', 'run', 'sample_round')]


tbl <- dat[Filename==x,][,list(
  Logloss_V = min(LogLoss_P1, na.rm=T),
  Logloss_H = min(LogLoss_H, na.rm=T), 
  main_task_V = main_task[which.min(LogLoss_P1)],
  main_task_H = main_task[which.min(LogLoss_H)],
  blueprint_V = Blueprint[which.min(LogLoss_P1)],
  blueprint_H = Blueprint[which.min(LogLoss_H)]
), by=c('Filename', 'run', 'sample_round')]

tbl <- dcast.data.table(tbl, Filename + sample_round ~ run, value.var=c('Logloss_V', 'Logloss_H', 'main_task_V', 'main_task_H'))
tbl

# XGB vs XGB
dat[Filename==x & sample_round == 64 & main_task=='ESXGBC' & run=='June_22',list(main_task, LogLoss_H)]

dat[Filename==x & sample_round == 64 & main_task=='ESXGBC' & run=='June_24',list(main_task, LogLoss_H)]

# Confirm blueprints are the same:
dat[Filename==x & sample_round == 64 & main_task=='ESXGBC' & run=='June_22', sort(Blueprint)] == dat[Filename==x & sample_round == 64 & main_task=='ESXGBC' & run=='June_24', sort(Blueprint)]

######################################################
# Get dataset
######################################################

url <- 'https://s3.amazonaws.com/datarobot_public_datasets/chargeback_clean_80.csv'
chargeback_raw <- fread(url)
chargeback <- copy(chargeback_raw)
setnames(chargeback, make.names(gsub('"', '', names(chargeback), fixed=T), unique = T))
chargeback[,postalCode := gsub('"', '', postalCode)]
chargeback[is.na(as.numeric(postalCode)),table(postalCode)]

# wget https://s3.amazonaws.com/datarobot_public_datasets/chargeback_clean_80.csv
# sed 's/None/NONENONENONENONE/g' chargeback_clean_80.csv > chargeback_clean_replace_none_80.csv
######################################################
# Table of results
######################################################

vars = c('Filename', 'Y_Type', 'variable', old_release, new_release, 'diff')
res_normal = res[variable == 'Gini_H' & abs(diff) > 0.01 & sample_round==64, vars, with=F]

values = c(old_release, new_release, 'diff')
res_normal = dcast.data.table(res_normal, Filename + Y_Type ~ variable, value.var = values)

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

values = c(old_release, new_release)
cat_norm = dcast.data.table(cat_norm, Filename ~ variable, value.var = values)

res_normal = merge(res_normal, cat_norm, by='Filename')[order(diff_Gini_H),]
res_normal

######################################################
# Lookit issues
######################################################

res[,pct_diff := (get(new_release) / get(old_release)) - 1]

res[sample_round==64 & diff > 0.1 & pct_diff > 0.25 & variable %in% c('Total_Time_P1_Hours', 'Max_RAM_GB'),][order(pct_diff),]

res[
  sample_round == 64 & variable %in% 'Total_Time_P1_Hours' & !is.na(diff),][which.max(diff),]

res[
  sample_round == 0 & variable %in% 'Total_Time_P1_Hours' & !is.na(diff),][which.min(diff),]

res[sample_round == 64 & variable %in% plot_vars & !is.na(diff) & diff > 1,]


x <- 'chargeback_clean_80.csv'
dat[Filename==x & run == old_release,][which.max(Max_RAM_GB), main_task]
dat[Filename==x & run == new_release,][which.max(Max_RAM_GB), main_task]
dat[Filename==x & run == old_release,][which.max(Total_Time_P1_Hours), main_task]
dat[Filename==x & run == new_release,][which.max(Total_Time_P1_Hours), main_task]

dat[Filename==x,][,list(
  Logloss_V=min(LogLoss_P1, na.rm=T),
  Logloss_H=min(LogLoss_H, na.rm=T)), by=c('Filename', 'sample_round')]

