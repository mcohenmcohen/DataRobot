stop()
rm(list=ls(all=T))
gc(reset=T)
library(yaml)
library(data.table)
library(readr)
library(httr)
library(pbapply)
library(stringi)
library(jsonlite)
library(ggplot2)
library(shrinkR)
library(reshape2)

# Regular Mbtest
dat_full <- fread('http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests=59d3abaa1c443f0001898bf7&max_sample_size_only=false')

# Dev MBP of tensorflow
#dat_tf <- fread('http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests=59d433fa29162300013c2d9c&max_sample_size_only=false')
dat_tf <- fread('http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests=59d6b2bc192eba0001a3384b&max_sample_size_only=false')

# Combine
dat <- rbind(dat_full[!main_task %in% c('TFNNC', 'LR1'),], dat_tf, fill=T)
dat <- dat[Filename != 'iris.csv',]
#dat <- cleanLeaderboard(dat)
#dat <- titleModels(dat)

# Munge args
dat[,args_json := stri_c('{"', main_args, '"}')]
dat[,args_json := stri_replace_all_fixed(args_json, ';', '","')]
dat[,args_json := stri_replace_all_fixed(args_json, '=', '": "')]
dat[,args_json := stri_replace_all_fixed(args_json, ',""', '')]
dat[,args_json := stri_replace_all_fixed(args_json, '{""}', '{"noargs": 1}')]
args_df <- dat[, rbindlist(sapply(args_json, fromJSON), fill=T)]
dat <- data.table(dat, args_df)

# Re-title NN models
dat[,model_name := main_task]
dat[!is.na(es), model_name := paste0(model_name, ' epoch_size=', es)]
dat[!is.na(ni), model_name := paste0(model_name, ' iters=', ni)]
#dat[,table(model_name)]

# Munge units
dat[,Max_RAM_GB := as.numeric(Max_RAM * 1e-9)]
dat[,Total_Time_P1_Hours := Total_Time_P1/3600]
dat[,size_GB := size * 1e-9]

# Summary
# x = dat[,list(main_task, Filename, Sample_Size, Sample_Pct, LogLoss_P1, Total_Time_P1_Hours, Max_RAM_GB, size_GB)]
# summary(x)
# x[Total_Time_P1_Hours > 2,][order(Total_Time_P1_Hours),]

# Bar charts
plotdat <- dat[main_task %in% c('ESXGBC2', 'TFNNC', 'DLNNC'),]
ggplot(plotdat, aes(x=model_name, y=LogLoss_P1)) + geom_boxplot() + theme_bw() + scale_y_log10()

# Scatterplots
ids <- c('model_name', 'Filename', 'Sample_Size', 'Blueprint')
measures <- c('LogLoss_P1', 'AUC_P1', 'Total_Time_P1_Hours', 'Max_RAM_GB', 'size_GB')
dat <- melt.data.table(dat[,c(ids, measures), with=F], id.vars=ids, measure.vars=measures)
dat <- dcast.data.table(dat, Filename + Sample_Size + variable ~ model_name)

# Deep learning
plotdat <- dat[!is.na(ESXGBC2) & !is.na(`DLNNC epoch_size=full iters=2000`),]
plotdat[, list(
  ESXGBC2 = median(ESXGBC2),
  DLNNC = median(`DLNNC epoch_size=full iters=2000`)
), by='variable']
ggplot(plotdat, aes(x=ESXGBC2, y=`DLNNC epoch_size=full iters=2000`)) +
  geom_point() + theme_bw() + geom_abline(slope=1, intercept=0) + facet_wrap(~variable, scales='free')

# Accuracy optimized MLP
plotdat <- dat[!is.na(ESXGBC2) & !is.na(`TFNNC epoch_size=full iters=2000`),]
plotdat[, list(
  ESXGBC2 = median(ESXGBC2),
  TFNNC_Full = median(`TFNNC epoch_size=full iters=2000`)
), by='variable']
ggplot(plotdat, aes(x=ESXGBC2, y=`TFNNC epoch_size=full iters=2000`)) +
  geom_point() + theme_bw() + geom_abline(slope=1, intercept=0) + facet_wrap(~variable, scales='free')

# Basic MLP
plotdat <- dat[!is.na(ESXGBC2) & !is.na(`TFNNC epoch_size=10000 iters=100`),]
plotdat[, list(
  ESXGBC2 = median(ESXGBC2),
  TFNNC_10k = median(`TFNNC epoch_size=10000 iters=100`)
), by='variable']
ggplot(plotdat, aes(x=ESXGBC2, y=`TFNNC epoch_size=10000 iters=100`)) +
  geom_point() + theme_bw() + geom_abline(slope=1, intercept=0) + facet_wrap(~variable, scales='free')

# Basic MLP vs Accuracy optimized MLP
plotdat <- dat[!is.na(`TFNNC epoch_size=full iters=2000`) & !is.na(`TFNNC epoch_size=10000 iters=100`),]
plotdat[, list(
  TFNNC_Full = median(`TFNNC epoch_size=full iters=2000`),
  TFNNC_10k = median(`TFNNC epoch_size=10000 iters=100`)
), by='variable']
ggplot(plotdat, aes(
  x=`TFNNC epoch_size=full iters=2000`,
  y=`TFNNC epoch_size=10000 iters=100`)) +
  geom_point() + theme_bw() + geom_abline(slope=1, intercept=0) + facet_wrap(~variable, scales='free')

