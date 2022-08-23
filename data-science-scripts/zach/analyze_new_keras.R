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

# Newer set of master Mbtests
# old_mbtest_ids <- c(
#   '5ce5a4b27347c9002707db2a',  # FM Yaml
#   '5ce5a44a7347c90029cc30a8',  # Current with preds Yaml
#   '5ce5a48c7347c9002707db20',  # Cosine sim
#   '5ce5a49d7347c900251e88fe'  # Single column text
# )

old_mbtest_ids <- '5d570a207347c900279268eb'  # New current with preds with other yamls added

# Keras tests also listed here: https://github.com/datarobot/DataRobot/pull/37647

# Keras models - current - broken passthrough - needs work
# new_mbtest_ids <- c(
#   '5ce5a3f87347c9002707db0c',  # FM Yaml
#   '5ce5a3807347c900245e4dfe',  # Current with preds Yaml
#   '5ce5a3b57347c900251e88ed',  # Cosine sim
#   '5ce5a3e17347c900245e4fe1'  # Single column text
# )

# Keras models - current - working passthrough - Looks ok
# new_mbtest_ids <- c(
#   '5ce5a3f87347c9002707db0c',  # FM Yaml
#   '5ce5a3807347c900245e4dfe',  # Current with preds Yaml
#   '5ce5a3b57347c900251e88ed',  # Cosine sim
#   '5ce5a3e17347c900245e4fe1'  # Single column text
# )

# # Keras models - trying to calibrate - CLOSE THE PR THESE ARE TOO SLOW
# new_mbtest_ids <- c(
#   '5ce5b6997347c90029cc32a9',  # FM Yaml
#   '5ce5b64b7347c9002707db42',  # Current with preds Yaml
#   '5ce5b6757347c90029cc329f',  # Cosine sim
#   '5ce5b6877347c900245e5001'  # Single column text
# )

# Keras models - current - working passthrough - fixed weight init for multiclass
# Best so far
# new_mbtest_ids <- c(
#   '5d02bf2f7347c90026e2d02c',  # FM Yaml
#   '5d02bef07347c9002931fa58',  # Current with preds Yaml
#   '5d02bf047347c900248d472a',  # Cosine sim
#   '5d02bf177347c90026e2d01a'  # Single column text
# )

# Keras models - current - working passthrough - fixed weight init for multiclass - learning rate = 1 - CURRENT TEST!
# learning rate = 1 is no good
# new_mbtest_ids <- c(
#   '5d0a357e7347c90027fba955',  # FM Yaml
#   '5d0900997347c90027fba662',  # Current with preds Yaml
#   '5d0900af7347c900284e1a1c',  # Cosine sim
#   '5d0900bf7347c900291de9eb'  # Single column text
# )

# Keras models - current - 0.1 for class, 0.01 for reg.  Find learning rate, cyclic lr, early stopping, smaller default batch size
# OOMS due to stacked predictions.  Jesse/Viktor working on RAM FIX
# new_mbtest_ids <- c(
#   '5d2659877347c9002610cd64',  # FM Yaml
#   '5d2659507347c9002c6234a0',  # Current with preds Yaml
#   '5d26595d7347c9002c62368f',  # Cosine sim
#   '5d2659757347c9002610cd51'  # Single column text
# )

# Keras models - current - 0.1 for class, 0.01 for reg.  Find learning rate, cyclic lr, early stopping, smaller default batch size
# RUN ALL AS SLIM RUN TO AVOID THE MULTI MODEL RAM ISSUE
# new_mbtest_ids <- c(
#   '5d31dac57347c90027198d85',  # FM Yaml
#   '5d31daa17347c90027198b97',  # Current with preds Yaml
#   '5d31dad57347c90029160106',  # Cosine sim
#   '5d31dae27347c90023a8bb3e'  # Single column text
# )

# New test, with just find learning rate turned on
# Some OOMs during pickling =/
# NOT SLIM
# new_mbtest_ids <- c(
#   '5d41c5057347c90029fd538b',  # FM Yaml
#   '5d41c4c37347c90025f7bd51',  # Current with preds Yaml
#   '5d41c4da7347c90025f7bf40',  # Cosine sim
#   '5d41c4eb7347c90025f7bf49'  # Single column text
# )
#
# # New test, with just find learning rate turned on - ACTUALLY SLIM NOW
# new_mbtest_ids <- c(
#   '5d4351067347c90026eeeb73',  # FM Yaml
#   '5d4350b57347c9002bf429a1',  # Current with preds Yaml
#   '5d4350dc7347c9002bf42b92',  # Cosine sim
#   '5d4350f27347c90026eeeb60'  # Single column text
# )

# New test, with just find learning rate turned on - ACTUALLY SLIM NOW, MIN BATCH OF 1
# new_mbtest_ids <- c(
#   '5d44a0a37347c9002bc9ed12',  # FM Yaml
#   '5d44a04a7347c900248b1639',  # Current with preds Yaml
#   '5d44a07b7347c900248b182a',  # Cosine sim
#   '5d44a0957347c9002bc9ecff'  # Single column text
# )

# # New test, just min batch of 1, no find lr
# new_mbtest_ids <- c(
#   '5d456e467347c90029d7283c',  # FM Yaml
#   '5d456dd57347c90025ac0cde',  # Current with preds Yaml
#   '5d456e2d7347c900248b1843',  # Cosine sim
#   '5d456e1c7347c90025ac0ecd'  # Single column text
# )

# Find LR + jason's fix + batch size 1
# new_mbtest_ids <- c(
#   '5d478b517347c9002b435680',  # FM Yaml
#   '5d478ab47347c90024ea9c6d',  # Current with preds Yaml - FAILED DEPLOY
#   '5d478b137347c9002b435664',  # Cosine sim - FAILED DEPLOY
#   '5d478b347347c9002b43566d'  # Single column text - FAILED DEPLOY
# )

# Find LR + min batch size of 1 + bug fix for small datasets with only 1 or 2 LR find epochs
# Basically a retest of the above
# new_mbtest_ids <- c(
#   '5d48bb307347c9002bb03933',  # FM Yaml
#   '5d48ba6d7347c90029a74b4b',  # Current with preds Yaml
#   '5d48bada7347c9002410c54e',  # Cosine sim
#   '5d48bafb7347c90029a74d3c'  # Single column text
# )

# Rerun of the above, because I thought they failed to deploy, but they didnt!
# new_mbtest_ids <- c(
#   '5d497fb57347c9002504c393',  # FM Yaml
#   '5d497f7d7347c9002b3e8fe4',  # Current with preds Yaml
#   '5d497f8f7347c9002a81cfaf',  # Cosine sim
#   '5d497f9d7347c9002a81cfb9'  # Single column text
# )

new_mbtest_ids <- '5d570a027347c900279266ef'  # Test with current with preds file, min lr / 10 heuristic + smaller batch size

# Name 'em
testnames <- c('Current With Preds')
names(old_mbtest_ids) <- testnames
names(new_mbtest_ids) <- testnames
all_tests <- c(old_mbtest_ids, new_mbtest_ids)

prefix = 'http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests='
suffix = '&max_sample_size_only=false'

# Read and name
read_and_name <- function(id){
  url <- paste0(prefix, id, suffix)
  out <- fread(url)
  out[,mbtest_id := id]
  out[,mbtest_name := names(all_tests[id == all_tests])]
  return(out)
}
dat_old_raw <- pblapply(old_mbtest_ids, read_and_name)
dat_new_raw <- pblapply(new_mbtest_ids, read_and_name)

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
  x[,x_prod_2_max_cardinal := NULL]
  return(x)
}

dat_old <- lapply(dat_old, clean_data)
dat_new <- lapply(dat_new, clean_data)

stopifnot(all(sapply(dat_old, function(x) 'Max_RAM_GB' %in% names(x))))
stopifnot(all(sapply(dat_new, function(x) 'Max_RAM_GB' %in% names(x))))

######################################################
# Combine data within each test
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

dat_old[,run := 'master']
dat_new[,run := 'keras']

stopifnot(dat_old[,all(Metablueprint=='Metablueprint v12.0.03-so')])
stopifnot(dat_new[,all(Metablueprint=='Test_Keras v2')])

######################################################
# Combine data BETWEEN the 2 tests
######################################################

tf_bps <- c('TFNNC', 'TFNNR')
keras_bps <- c('KERASR', 'KERASC', 'KERASMULTIC')
nn_bps <- c(tf_bps, keras_bps)

# Subset to RF only
# dat_old <- dat_old[main_task %in% c('RFC', 'RFR'),]

# Exclude baseline BPs from the keras MBtest
dat_new <- dat_new[main_task %in% keras_bps,]

# Combine into 1
dat <- rbindlist(list(dat_old, dat_new), use.names=T)

# Map names to test
filename_to_test_map <- unique(dat[,list(Filename, mbtest_name)])
filename_to_test_map <- filename_to_test_map[!duplicated(Filename),]

######################################################
# Add some vars
######################################################

dat[,dataset_bin := cut(dataset_size_GB, unique(c(0, 1.5, 2.5, 5, ceiling(max(dataset_size_GB)))), ordered_result=T, include.lowest=T)]
dat[,sample_round := Sample_Pct]
dat[sample_round=='--', sample_round := '0']
dat[,sample_round := round(as.numeric(sample_round))]

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
dat[,main_args_list := split_to_named_list(main_args)]
dat[,loss := sapply(main_args_list, '[', 'loss')]
dat[,epochs := as.integer(sapply(main_args_list, '[', 'epochs'))]
dat[,hidden_units := sapply(main_args_list, '[', 'hidden_units')]
dat[,hidden_activation := sapply(main_args_list, '[', 'hidden_activation')]
dat[,learning_rate := as.numeric(sapply(main_args_list, '[', 'learning_rate'))]
dat[,batch_size := sapply(main_args_list, '[', 'batch_size')]
dat[,double_batch_size := sapply(main_args_list, '[', 'double_batch_size')]
dat[,scale_target := sapply(main_args_list, '[', 'scale_target')]
dat[,log_target := sapply(main_args_list, '[', 'log_target')]
dat[,table(hidden_units)]  # Get rid of list(512,64,64,64)

# ATM the prelu BPs look better
dat <- dat[hidden_activation == 'prelu' | is.na(hidden_activation),]
dat[,table(hidden_activation, useNA = 'always')]

######################################################
# Exclude some rows
######################################################

dat <- dat[which(!is_blender),]  # Exclude blenders to see if Keras will help blends
dat <- dat[which(!is_prime),]  # Exclude primes to see if Keras will help primes

# Exclude runs above 64%, as we only trained TF up to validation, and did not use the holdout
# TODO: exclude by autopilot round number
dat <- dat[sample_round <= 64,]

# Subset to one keras BP
# This is the "autopilot model"
dat <- dat[hidden_units %in% c('list(512)', '', NA),]

######################################################
# Summarize stats - non multiclass
######################################################

# Find a var
# a=sort(names(dat)); a[grepl('Y_Type', tolower(a))]
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
  MASE_V = min(`MASE_P1`),
  LogLoss_H = min(`LogLoss_H`),
  LogLoss_V = min(`LogLoss_P1`)
), by=c('run', 'Filename', 'Y_Type')]

measures = c(
  'Max_RAM_GB', 'Total_Time_P1_Hours', 'Gini_V', 'Gini_H', 'Gini_P', 'MASE_H', 'MASE_V', 'LogLoss_H', 'LogLoss_V')
for(v in measures){
  tmp = sort(unique(res[[v]]))
  wont_convert = !is.finite(as.numeric(tmp))
  if(any(wont_convert)){
    print(tmp[wont_convert])
  }
  set(res, j=v, value=as.numeric(res[[v]]))
}

res = melt.data.table(res, measure.vars=intersect(names(res), measures))
res = dcast.data.table(res, Filename + Y_Type + variable ~ run, value.var='value')

res[,diff := as.numeric(keras) - as.numeric(master)]

# Add test name
N <- nrow(res)
res <- merge(res, filename_to_test_map, all.x=T, by=c('Filename'))
stopifnot(N == nrow(res))

######################################################
# Plot of results - non multiclass
######################################################

plot_vars = c('Max_RAM_GB', 'Total_Time_P1_Hours', 'Gini_V', 'Gini_H')
plotdat <- res[
  variable %in% plot_vars & !is.na(keras) & !is.na(master),]
ggplot(plotdat, aes(x=`master`, y=`keras`, color=mbtest_name)) +
  geom_point() + geom_abline(slope=1, intercept=0) +
  facet_wrap(~variable, scales='free') +
  theme_bw() + theme_tufte() + ggtitle('keras vs master results')

res[keras > 5+master & variable=='Max_RAM_GB',]
res[keras > 1+master & variable=='Total_Time_P1_Hours',]

# Look for good demos
a=res[order(diff),][variable == 'Gini_V' & !is.na(diff) & diff>=0,]
b=res[order(diff),][variable == 'Gini_H' & !is.na(diff) & diff>=0,]
c=res[order(diff),][variable == 'Total_Time_P1_Hours' & !is.na(diff) & keras<0.09,]
x=merge(a, b, by=c('Filename', 'Y_Type'), all=F)
x=merge(x, c, by=c('Filename', 'Y_Type'), all=F)
x[,diff := (diff.x + diff.y)/2]
x[order(diff),][!is.na(diff),]
res[Filename=='reuters_text_train_80.csv',]

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

dat_nn <- dat[main_task %in% nn_bps,]
dat_nn[,table(main_task)]

res_nn <- copy(dat_nn)
res_nn <- res_nn[!is.na(Max_RAM_GB),]
res_nn <- res_nn[!is.na(Total_Time_P1_Hours),]
res_nn <- res_nn[!is.na(`Gini Norm_H`),]

# Repo models
#res_nn <- res_nn[(main_task %in% tf_bps) | (hidden_units == 'list(512 ,64, 64)'),]

# Autopilot models
res_nn <- res_nn[(main_task %in% tf_bps) | (hidden_units == 'list(512)'),]

res_nn <- res_nn[,list(
  Max_RAM_GB = max(Max_RAM_GB),
  Total_Time_P1_Hours = max(Total_Time_P1_Hours),
  Gini_V = max(`Gini Norm_P1`),
  Gini_H = max(`Gini Norm_H`),
  Gini_P = max(`Prediction Gini Norm`),
  MASE_H = min(`MASE_H`),
  MASE_V = min(`MASE_P1`),
  LogLoss_H = min(`LogLoss_H`),
  LogLoss_V = min(`LogLoss_P1`)
), by=c('run', 'Filename', 'Y_Type')]

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

# Runtime and RAM worse, but gini better
plot_vars = c('Max_RAM_GB', 'Total_Time_P1_Hours', 'Gini_V', 'Gini_H')
ggplot(res_nn[variable %in% plot_vars,], aes(x=master, y=keras, color=Y_Type)) +
  geom_point() + geom_abline(slope=1, intercept=0) +
  facet_wrap(~variable, scales='free') +
  theme_bw() + theme_tufte() + ggtitle('keras vs tensorflow results')

# Logloss worse
plot_vars = c('Max_RAM_GB', 'Total_Time_P1_Hours', 'LogLoss_V', 'LogLoss_H')
ggplot(res_nn[variable %in% plot_vars,], aes(x=master, y=keras, color=Y_Type)) +
  geom_point() + geom_abline(slope=1, intercept=0) +
  facet_wrap(~variable, scales='free') +
  theme_bw() + theme_tufte() + ggtitle('keras vs tensorflow results')

plot_vars = c('Max_RAM_GB', 'Total_Time_P1_Hours', 'Gini_V', 'Gini_H')
ggplot(res_nn[variable %in% plot_vars,]) +
  geom_density(aes(x=master), col='red', adjust=1.5) +
  geom_density(aes(x=keras), col='blue', adjust=1.5) +
  facet_wrap(~variable, scales='free') +
  theme_bw() + theme_tufte() + ggtitle('keras vs tensorflow results')
# Performs better in cases where NN Bps do better

######################################################
# Plot of results - multiclass - good results!
######################################################
plot_vars = c('Max_RAM_GB', 'Total_Time_P1_Hours', 'LogLoss_V', 'LogLoss_H')
ggplot(res[variable %in% plot_vars & Y_Type == 'Multiclass',], aes(x=`master`, y=`keras`)) +
  geom_point() + geom_abline(slope=1, intercept=0) +
  facet_wrap(~variable, scales='free') +
  theme_bw() + theme_tufte() + ggtitle('keras vs master results')

######################################################
# Worst logloss
######################################################

# Seems like the LR finder helps for text datasets
# LR finder sucks for 250p_PA_HS_3_years_since_debut_predict_70p_80.csv
# 0.89824 with find LR, 0.13497 without
# https://s3.amazonaws.com/datarobot_public_datasets/250p_PA_HS_3_years_since_debut_predict_70p_80.csv

res_nn[variable=='LogLoss_H' & Y_Type == 'Binary',][order(diff, decreasing=T),][1:10,]
# Filename Y_Type  variable   keras  master    diff
# 1:    250p_PA_HS_3_years_since_debut_predict_70p_80.csv Binary LogLoss_H 0.89824 0.12774 0.77050
# 2:                          DR_Demo_Telecomms_Churn.csv Binary LogLoss_H 0.87429 0.26408 0.61021
# 3:                        subreddit_text_cosine_sim.csv Binary LogLoss_H 1.09619 0.58165 0.51454
# 4:                                DR_Demo_AML_Alert.csv Binary LogLoss_H 0.74326 0.25443 0.48883
# 5:                                bio_grid_small_80.csv Binary LogLoss_H 0.67097 0.22656 0.44441
# 6: 28_Features_split_train_converted_train80_CVTVH3.csv Binary LogLoss_H 0.57519 0.13606 0.43913
# 7:   mlcomp1438_derivation-stats-balanced2_train_80.csv Binary LogLoss_H 1.01814 0.60560 0.41254
# 8:                                      Benefits_80.csv Binary LogLoss_H 0.92692 0.58602 0.34090
# 9:                                         wells_80.csv Binary LogLoss_H 1.00125 0.66479 0.33646
# 10:                            bio_exp_wide_train_80.csv Binary LogLoss_H 0.90703 0.59035 0.31668

res_nn[variable=='LogLoss_H' & Y_Type == 'Multiclass',][order(diff, decreasing=T),][1:10,]
# Filename     Y_Type  variable   keras  master    diff
# 1:                       mfeat-zernike_v1_80.csv Multiclass LogLoss_H 1.27910 0.39709 0.88201
# 2:                                          long Multiclass LogLoss_H 0.90126 0.36900 0.53226
# 3:                 weighted_rental_train_TVH.csv Multiclass LogLoss_H 0.50198 0.20268 0.29930
# 4:         GesturePhaseSegmentationRAW_v1_80.csv Multiclass LogLoss_H 1.20242 0.90726 0.29516
# 5:    weighted_and_dated_rental_train_TVH_80.csv Multiclass LogLoss_H 0.51190 0.21750 0.29440
# 6:                   internet_usage_v1_train.csv Multiclass LogLoss_H 2.24563 1.97423 0.27140
# 7:         10MB_downsampled_BNG(autos)_v1_80.csv Multiclass LogLoss_H 0.99556 0.73422 0.26134
# 8:                      JapaneseVowels_v1_80.csv Multiclass LogLoss_H 0.32340 0.06428 0.25912
# 9:  10MB_downsampled_BNG(autos,5000,5)_v1_80.csv Multiclass LogLoss_H 1.21086 0.95326 0.25760
# 10: 10MB_downsampled_BNG(autos,10000,1)_v1_80.csv Multiclass LogLoss_H 0.91936 0.68164 0.23772
# "long" is 0MB_downsampled_Physical_Activity_Recognition_Dataset_Using_Smartphone_Sensors_v1_80.csv

######################################################
# Worst runtime
######################################################

res_nn[variable=='Total_Time_P1_Hours' & Y_Type == 'Binary',][order(diff, decreasing=T),][1:10,]
res_nn[variable=='Total_Time_P1_Hours' & Y_Type == 'Multiclass',][order(diff, decreasing=T),][1:10,]

######################################################
# Worst runtime - overall
######################################################

res[variable=='Total_Time_P1_Hours',][order(diff, decreasing=T),][1:10,]

######################################################
# datasets to test
######################################################

dat[Filename=='quora_80.csv' & main_task == 'KERASC',Blueprint]

# [1] "{u'1': [[u'TXT'], [u'PTM3 a=word;b=1;d1=2;d2=0.5;dtype=float32;id=0;lc=1;maxnr=2;minnr=1;mxf=200000;n=l2;sw=None'], u'T'], u'2': [[u'1'], [u'KERASC batch_size=4096;double_batch_size=1;epochs=4;hidden_activation=prelu;hidden_units=list(512);learning_rate=0.01;loss=binary_crossentropy;max_batch_size=131072;pass_through_inputs=1;t_m=LogLoss'], u'P']}"


# https://s3.amazonaws.com/datarobot_public_datasets/quora_80.csv
# https://s3.amazonaws.com/datarobot_public_datasets/amazon_small_80.csv

# - dataset_name: https://s3.amazonaws.com/datarobot_public_datasets/ClickPrediction80.csv
#   metric: Tweedie Deviance
#   target: clicks
#
# - dataset_name: https://s3.amazonaws.com/datarobot_public_datasets/OnCampusArrests_80.csv
#   metric: Tweedie Deviance
#   target: LIQUOR12
#
# - dataset_name: https://s3.amazonaws.com/datarobot_public_datasets/cemst-decision-prediction2-asr3_train_80.csv
#   metric: LogLoss
#   target: y
#
# - dataset_name: https://s3.amazonaws.com/datarobot_public_datasets/trainingDataWithoutNegativeWeights_80.csv
#   metric: LogLoss
#   target: classification
#
# - dataset_name: https://s3.amazonaws.com/datarobot_public_datasets/bio_response_combined_80.csv
#   metric: LogLoss
#   target: Activity
#
# - dataset_name: https://s3.amazonaws.com/datarobot_public_datasets/bio_exp_wide_train_80.csv
#   target: regulated
#   metric: LogLoss
#
# - dataset_name: https://s3.amazonaws.com/datarobot_public_datasets/Gamblers_80.csv
#   metric: LogLoss
#   target: YES_ALCOHOL
