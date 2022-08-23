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
  '5d5add637347c90025d554a7'  # Zach's best keras on master
)
new_mbtest_ids <- c(
  '5d78a8b87347c90025c55126'  # Jason's new heuristics
)

old_release <- 'master'
new_release <- 'keras'
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

stopifnot(dat_old[,all(Metablueprint=='Metablueprint v12.0.03-so' | Metablueprint=='')])
stopifnot(dat_new[,all(Metablueprint=='Test_Keras v2')])

######################################################
# Add some BP info to keras tasks
######################################################

split_to_named_list <- function(x){
  out <- stri_split_fixed(x, ';')
  out <- lapply(out, function(a){
    tmp <- stri_split_fixed(a, '=')
    out <- sapply(tmp, '[', 2)
    names(out) <- sapply(tmp, '[', 1)
    return(out)
  })
  return(out)
}
dat_new[,main_args_list := split_to_named_list(main_args)]
dat_new[,loss := sapply(main_args_list, '[', 'loss')]
dat_new[,epochs := as.integer(sapply(main_args_list, '[', 'epochs'))]
dat_new[,hidden_units := sapply(main_args_list, '[', 'hidden_units')]
dat_new[,hidden_activation := sapply(main_args_list, '[', 'hidden_activation')]
dat_new[,learning_rate := as.numeric(sapply(main_args_list, '[', 'learning_rate'))]
dat_new[,batch_size := sapply(main_args_list, '[', 'batch_size')]
dat_new[,double_batch_size := sapply(main_args_list, '[', 'double_batch_size')]
dat_new[,scale_target := sapply(main_args_list, '[', 'scale_target')]
dat_new[,log_target := sapply(main_args_list, '[', 'log_target')]
dat_new[,table(hidden_units)]  # Get rid of list(512,64,64,64)

# ATM the prelu BPs look better
dat_new <- dat_new[hidden_activation == 'prelu' | is.na(hidden_activation),]
dat_new[,table(hidden_activation, useNA = 'always')]

#Print all BPs in the test
unique(dat_new[,list(hidden_units, learning_rate)])

# Subset to one keras BP
# This is the "autopilot model"
dat_new <- dat_new[hidden_units == 'list(128,512)',]
dat_new <- dat_new[learning_rate == 0.01,]

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
res = dcast.data.table(res, Filename + Y_Type + sample_round + variable ~ run, value.var='value')   # + main_task

set(res, j='diff', value = res[[new_release]] - res[[old_release]])
# res[abs(diff) < 0.01 & variable == 'Gini_H' & (sample_round == 0 | sample_round == 64),]

######################################################
# Plot of results
######################################################

plot_vars = c('Max_RAM_GB', 'Total_Time_P1_Hours', 'Logloss_H', 'Gini_H')
ggplot(res[sample_round == 64 & variable %in% plot_vars & !is.na(diff),], aes_string(x=old_release, y=new_release, color='Y_Type')) + 
  geom_point() + geom_abline(slope=1, intercept=0) + 
  facet_wrap(~variable, scales='free') + 
  theme_bw() + theme_tufte() + ggtitle(paste(test, old_release, 'vs', new_release, 'non time series results'))

######################################################
# Table of results - non multiclass
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

head(res_normal[order(diff_Gini_H),])

######################################################
# Table of results - non multiclass
######################################################

vars = c('Filename', 'Y_Type', 'variable', old_release, new_release, 'diff')
res_normal = res[variable == 'Logloss_H' & abs(diff) > 0.01 & sample_round==64, vars, with=F]

values = c(old_release, new_release, 'diff')
res_normal = dcast.data.table(res_normal, Filename + Y_Type ~ variable, value.var = values)

res_cat <- copy(dat)
res_cat <- res_cat[!is.na(Max_RAM_GB),]
res_cat <- res_cat[!is.na(Total_Time_P1_Hours),]
res_cat <- res_cat[!is.na(LogLoss_H),]

res_cat <- res_cat[,list(
  best_logloss = main_task[which.min(LogLoss_H)]
), by=c('run', 'Filename', 'sample_round')]

measures = c('best_logloss', 'best_mase_model')
res_cat = melt.data.table(res_cat, measure.vars=intersect(names(res_cat), measures))
res_cat = dcast.data.table(res_cat, Filename + sample_round + variable ~ run, value.var='value')

cat_norm = res_cat[sample_round==64 & variable == 'best_logloss',]

values = c(old_release, new_release)
cat_norm = dcast.data.table(cat_norm, Filename ~ variable, value.var = values)

res_normal = merge(res_normal, cat_norm, by='Filename')[order(diff_Logloss_H),]

tail(res_normal[order(diff_Logloss_H),])

######################################################
# Table of results - holdout - non multiclass
######################################################
# Holdout is 20%, so is a larger sample to compare on
# Valid should be good too, as we're comparing up to 64% only.

res_normal = res[variable == 'Gini_H' & diff >= 0,
                 list(Filename, Y_Type, variable, `master`, `keras`, diff)]
values = c('master', 'keras', 'diff')
res_normal = dcast.data.table(res_normal, Filename + Y_Type ~ variable, value.var = values)

res_cat <- copy(dat)
res_cat <- res_cat[!is.na(Max_RAM_GB),]
res_cat <- res_cat[!is.na(Total_Time_P1_Hours),]
res_cat <- res_cat[!is.na(`Gini Norm_H`),]

res_cat <- res_cat[,list(
  best_gini_model = main_task[which.max(`Gini Norm_H`)],
  best_mase_model = main_task[which.min(MASE_H)]
), by=c('run', 'Filename')]

measures = c('best_gini_model', 'best_mase_model')
res_cat = melt.data.table(res_cat, measure.vars=intersect(names(res_cat), measures))
res_cat = dcast.data.table(res_cat, Filename + variable ~ run, value.var='value')

cat_norm = res_cat[variable == 'best_gini_model',]

values = c('master', 'keras')
cat_norm = dcast.data.table(cat_norm, Filename ~ variable, value.var = values)

res_normal = merge(res_normal, cat_norm, by='Filename')[order(diff_Gini_H),]

# HUGE improvement on single column text datasets
# HUGE improvements on cosine similarity
# MASSIVELY HUGE improvement on xor text dataset
res_normal[order(diff_Gini_H),]

# On about 8.9%% of datasets, better than the best blender on master!
res[!is.na(diff) & variable == 'Gini_V', sum(diff > 0) / .N]
res[!is.na(diff) & variable == 'Gini_H', sum(diff > 0) / .N]

######################################################
# Compare to old TF Bps
######################################################

tf_bps <- c('TFNNC', 'TFNNR')
keras_bps <- c("KERASC", "KERASMULTIC", "KERASR")
nn_bps <- c(tf_bps, keras_bps)

dat_new[,table(main_task)]
dat_nn <- dat[main_task %in% nn_bps,]
dat_nn[,table(main_task)]

res_nn <- copy(dat_nn)
res_nn <- res_nn[!is.na(Max_RAM_GB),]
res_nn <- res_nn[!is.na(Total_Time_P1_Hours),]
res_nn <- res_nn[!is.na(`Gini Norm_H`),]

# Repo models
#res_nn <- res_nn[(main_task %in% tf_bps) | (hidden_units == 'list(512 ,64, 64)'),]

# Autopilot models
res_nn <- res_nn[(main_task %in% tf_bps) | (hidden_units == 'list(128,512)'),]

res_nn <- res_nn[,list(
  Max_RAM_GB = max(Max_RAM_GB),
  Total_Time_P1_Hours = max(Total_Time_P1_Hours),
  Gini_V = max(`Gini Norm_P1`),
  Gini_H = max(`Gini Norm_H`),
  LogLoss_H = min(`LogLoss_H`),
  LogLoss_V = min(`LogLoss_P1`)
), by=c('run', 'Filename', 'Y_Type')]
res_nn[,table(run)]

measures = c(
  'Max_RAM_GB', 'Total_Time_P1_Hours', 'Gini_V', 'Gini_H', 'Gini_P', 'MASE_H', 'MASE_V', 'LogLoss_H', 'LogLoss_V')
res_nn = melt.data.table(res_nn, measure.vars=intersect(names(res_nn), measures))
res_nn = dcast.data.table(res_nn, Filename + Y_Type + variable ~ run, value.var='value')
res_nn[,keras := as.numeric(`keras`)]
res_nn[,master := as.numeric(`master`)]
res_nn[,diff := keras - master]

# Table by gini - V
# 80% better
# trainingDataWithoutNegativeWeights_80.csv
# DR_Demo_Pred_Main_Reg.csv
# terror_mix_train_80.csv
# New_York_Mets_Ian_11.csv
# ofnp_80.csv
summary(res_nn[variable == 'Gini_V',])
res_nn[variable == 'Gini_V'][order(diff),][1:5,]
res_nn[variable == 'Gini_V' & !is.na(diff), sum(diff >= 0) / .N]

# Table by gini - H
# 76% better
# trainingDataWithoutNegativeWeights_80.csv
# DR_Demo_Pred_Main_Reg.csv
# New_York_Mets_Ian_11.csv
summary(res_nn[variable == 'Gini_H',])
res_nn[variable == 'Gini_H'][order(diff),][1:5,]
res_nn[variable == 'Gini_H' & !is.na(diff), sum(diff >= 0) / .N]

# Table by logloss - V
# Worst diff very large
# Best diff large
# Too many epochs?  Early stopping?  Weight decay?
# Gamblers_80.csv > 3.5 logloss diff!
# trainingDataWithoutNegativeWeights_80.csv > 3.5 logloss diff!
summary(res_nn[variable == 'LogLoss_V',])
res_nn[variable == 'LogLoss_V'][order(-diff),][1:5,]
res_nn[variable == 'LogLoss_V' & !is.na(diff), sum(diff <= 0) / .N]

# Table by logloss - H
# Too many epochs?  Early stopping?  Weight decay?
# Gamblers_80.csv > 3.5 logloss diff!
# trainingDataWithoutNegativeWeights_80.csv > 3.5 logloss diff!
summary(res_nn[variable == 'LogLoss_H',])
res_nn[variable == 'LogLoss_H'][order(-diff),][1:5,]
res_nn[variable == 'LogLoss_H' & !is.na(diff), sum(diff <= 0) / .N]
######################################################
# Lookit issues
######################################################

# Runtime/ram issues
res[,pct_diff := (get(new_release) / get(old_release)) - 1]
res[sample_round==64 & diff > 0.1 & pct_diff > 0.25 & variable %in% c('Total_Time_P1_Hours', 'Max_RAM_GB'),][order(pct_diff),]
