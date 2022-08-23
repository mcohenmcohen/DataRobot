######################################################
# Setup
######################################################

library(data.table)
library(bit64)
library(ggplot2)
library(Hmisc)

######################################################
# Download data
######################################################

dat_raw_1 <- fread('http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests=5941994e82f5af383df04df9&max_sample_size_only=false')
dat_raw_2 <- fread('http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests=5939b0e280c44010ce58ed8d&max_sample_size_only=false')
bench_raw <- fread('http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests=591f3951dcd4cb4c42f92271&max_sample_size_only=false')
dat_5gb_full <- fread('http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests=594934af5890cd26b3a169ca&max_sample_size_only=false')

######################################################
# Organize data
######################################################

benchmark <- copy(bench_raw)
benchmark[grepl('allstate', Filename), table(Filename)]
benchmark[grepl('allstate', Filename), summary(`Gini Norm_H`)]

dat <- rbindlist(list(dat_raw_1, dat_raw_2, dat_5gb_full), fill=T)
#dat <- copy(dat_5gb_full)

dat[,table(Filename)]
dat[is.na(dataset_size),table(Filename)]

dat[,Max_RAM_GB := as.numeric(Max_RAM * 1e-9)]
dat[,summary(Max_RAM_GB)]

dat[,Total_Time_P1_Hours := Total_Time_P1/3600]
dat[,summary(Total_Time_P1_Hours)]
dat[,summary(`Gini Norm_P1`)]

dat[,size_GB := size * 1e-9]
dat[,summary(size_GB)]
dat[,unique(data.frame(Filename, size_GB)[order(size_GB),])]

dat[,dataset_size_GB := as.numeric(dataset_size * 1e-9)]
dat[,summary(dataset_size_GB)]

dat[,dataset_bin := cut(dataset_size_GB, unique(c(0, 1.5, 2.5, 5, ceiling(max(dataset_size_GB)))), ordered_result=T, include.lowest=T)]
dat[,table(dataset_bin, useNA = 'ifany')]

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
