######################################################
# Setup
######################################################

library(data.table)
library(bit64)
library(ggplot2)
library(Hmisc)
library(jsonlite)
library(reshape2)
library(stringi)
library(ggplot2)

# TODO: Compare keras results to DR results

######################################################
# Download data
######################################################

dat_raw1 = fread('http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests=5b4e62777347c900278f8192&max_sample_size_only=false')
dat_raw2 = fread('http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests=5b4f65537347c900207c5e26&max_sample_size_only=false')

dat_keras = fread('http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests=5b9666cb7347c9002a0d4d79&max_sample_size_only=false')

######################################################
# Organize data
######################################################

dat_old <- rbindlist(list(dat_raw1, dat_raw2), use.names=T, fill=T)
#dat_old <- copy(dat_raw1)
dat_new <- copy(dat_keras)[main_task %in% c('KERASC', 'KERASR'),]

dat_old[,run := 'DR']
dat_new[,run := 'Keras']

dat <- rbindlist(list(dat_old, dat_new), use.names=T)

#dat[,table(Filename)]
dat[is.na(dataset_size),table(Filename)]

dat[,Max_RAM_GB := as.numeric(Max_RAM * 1e-9)]
dat[,summary(Max_RAM_GB)]

dat[,Total_Time_P1_Hours := Total_Time_P1/3600]
dat[,summary(Total_Time_P1_Hours)]
dat[,summary(`Gini Norm_P1`)]

dat[,size_GB := size * 1e-9]
dat[,summary(size_GB)]
#dat[,unique(data.frame(Filename, size_GB)[order(size_GB),])]

dat[,dataset_size_GB := as.numeric(dataset_size * 1e-9)]
#dat[,summary(dataset_size_GB)]

dat[,dataset_bin := cut(dataset_size_GB, unique(c(0, 1.5, 2.5, 5, ceiling(max(dataset_size_GB)))), ordered_result=T, include.lowest=T)]
#dat[,table(dataset_bin, useNA = 'ifany')]

######################################################
# Remove blenders
######################################################
dat <- dat[which(!is_blender),]

######################################################
# Ran with diff mercari datasets
######################################################
dat[Filename == 'mer_train_large.csv', Filename := 'mer_text_combo.csv']

dat[Filename == 'mer_text_combo.csv',][which.min(RMSLE_H + RMSLE_P1),list(RMSLE_P1, main_args)]
best_bp = 'loss=mean_absolute_error;epochs=4;hidden_units=list(512,64,64);activation=prelu;learning_rate=0.003;batch_size=4096;double_batch_size=True;scale_target=True;log_target=True;'
dat[main_args == best_bp,summary(`Gini Norm_P1`)]


######################################################
# Analyze keras improvement
######################################################
dat[,summary(`Gini Norm_P1`)]
dat[,summary(`Gini Norm_H`)]
dat[,gini := `Gini Norm_P1`]

# Gini - keras wins on some datasets!
tbl <- dat[!is.na(gini),list(gini=max(gini)), by=c('Filename', 'run')][order(gini),]
tbl <- dcast.data.table(tbl, Filename ~ run)
tbl[,diff := Keras/DR]
tbl[order(diff),]

# time - keras is fasssssssst
tbl <- dat[!is.na(Total_Time_P1_Hours),list(time=max(Total_Time_P1_Hours)*60), by=c('Filename', 'run')][order(time),]
tbl <- dcast.data.table(tbl, Filename ~ run)
tbl[,diff := Keras/DR]
tbl[order(diff),]

######################################################
# Analyze keras best settings
######################################################
keras = dat[run == 'Keras' & main_task %in% c('KERASC', 'KERASR'),]
keras = keras[grepl('mean_absolute_error', main_args, fixed=T),]
keras[,ridit := grepl('RDT5', `_tasks`)]
keras[,st := grepl('RST', `_tasks`)]
keras[,table(ridit, st)]

# Question 1: ridit or std?
# No big diff, ridit slightly better
keras[ridit == TRUE, summary(gini)]
keras[st == TRUE, summary(gini)]
keras[,best := 1:.N == which.max(gini), by='Filename']
keras[best==TRUE,table(ridit, st)]
ggplot(keras, aes(x=ridit, y=gini)) + geom_boxplot() + theme_bw()
ggplot(keras, aes(x=st, y=gini)) + geom_boxplot() + theme_bw()

# Question 2: layers and tuning!
#keras[best==TRUE,table(main_args)]
keras = keras[ridit == TRUE | (ridit == FALSE & st == FALSE)]
keras[,main_args := gsub('loss=[a-z_]+;', '', main_args)]
keras[,main_args := stri_trim_both(main_args)]
tbl2 = keras[,list(
  min = min(gini),
  mean = mean(gini),
  med = median(gini),
  max = max(gini)
  ), by='main_args'][order(med),]
tbl2[which.max(min),]
tbl2[which.max(med),]
tbl2[which.max(mean),]
tbl2[which.max(max),]

# More epochs seems better.  Maybe test 5?
# hidden_units=list(512) and learning_rate=0.01 (shallow fast)
# hidden_units=list(512,64,64) and 0.001 (deep slow)
# Use prelu
# CHECK THESE BPS

# Lookit those BPs
tbl2[main_args == 'epochs=4;hidden_units=list(512);activation=prelu;learning_rate=0.01;batch_size=4096;double_batch_size=True;scale_target=True;',]
tbl2[main_args == 'epochs=3;hidden_units=list(512,64,64);activation=prelu;learning_rate=0.01;batch_size=4096;double_batch_size=True;scale_target=True;',]



loss=mean_absolute_error;epochs=4;hidden_units=list(512,64,64);activation=prelu;learning_rate=0.003;batch_size=4096;double_batch_size=True;scale_target=True

mean_absolute_error



######################################################
# Analyze - preds - OLD CODE
######################################################

# Find prediction scores
dat[, metric := gsub('Weighted ','',  metric, fixed=T)]
for(m in dat[,sort(unique(metric))]){
  dat[metric == m, error_pred := get(paste('Prediction', metric))]
}

# Subset to max sample size
dat[,largest := Sample_Pct == max(Sample_Pct), by='Filename']
dat <- dat[largest==TRUE,]

# Subset to files with predictions
dat <- dat[!is.na(`Prediction dataset_name`),]

# Subset to only data with predicitons
dat <- dat[!is.na(error_pred),]

# Choose best model based on holdout
dat[,best := 1:.N==which.min(`error_H`), by='Filename']
dat <- dat[best==TRUE,]

# Subset columns
dat <- dat[,sapply(dat, function(x) length(unique(x))) > 1, with=F]
#dat[Filename=='criteo_20pct_in_mem_5GB.csv', ]
#dat[Filename=='newspaper_articles_train.csv',]

# Print
out <- dat[,list(Filename, metric, main_task, error_pred)]
out <- out[order(metric, error_pred),]
print(out)
fwrite(out, '~/Downloads/out.csv')
















######################################################
# OLDER CODE!  WORKS BUT THE ANALYSIS ISNT AS VALUABLE
######################################################














######################################################
# Regression
######################################################

######################################################
# Classification
######################################################

######################################################
# Multiclass
######################################################

######################################################
# Inspect
######################################################

dat[,unique(data.frame(Filename, size_GB)[order(size_GB),])]
dat[,list(`Gini.Norm` = max(`Gini Norm_H`, na.rm=T)), by='Filename'][order(Gini.Norm),]

dat[is.na(Max_RAM_GB),table(Filename)]
dat[is.na(Max_RAM_GB),table(pid)]


dat[which.max(Max_RAM_GB),table(Filename)]
dat[which.max(Max_RAM_GB),table(pid)]

dat[,list(.N), by='pid'][which.max(N),]

######################################################
# Bar charts
######################################################

dat[main_task != 'RULEFITR',][which.max(Max_RAM_GB),list(Filename, main_task, Max_RAM_GB)]
dat[which.max(Total_Time_P1_Hours),list(Filename, main_task, Total_Time_P1_Hours)]

#RAM
ggplot(dat, aes(x=dataset_bin, y=Max_RAM_GB)) +
  geom_boxplot() +
  theme_bw() +
  ggtitle('Max RAM usage by Dataset Size') +
  xlab('Dataset Size (GB) + 95% CI') +
  ylab('Max RAM Usage')

#Runtime
ggplot(dat, aes(x=dataset_bin, y=Total_Time_P1_Hours)) +
  geom_boxplot() +
  theme_bw() +
  ggtitle('Runtime by Dataset Size') +
  xlab('Dataset Size (GB)') +
  ylab('Partition 1 Runtime (Hours) + 95% CI')

#Accuracy
ggplot(dat, aes(x=dataset_bin, y=`Gini Norm_P1`)) +
  geom_boxplot() +
  theme_bw() +
  ggtitle('Accuracy by Dataset Size') +
  xlab('Dataset Size (GB)') +
  ylab('Partition 1 Gini Norm + 95% CI')

######################################################
# Scatter plots
######################################################

#RAM
ggplot(dat, aes(x=dataset_size_GB, y=Max_RAM_GB)) +
  geom_point() +
  theme_bw() +
  ggtitle('Max RAM usage by Dataset Size') +
  xlab('Dataset Size (GB)') +
  ylab('Max RAM Usage')
