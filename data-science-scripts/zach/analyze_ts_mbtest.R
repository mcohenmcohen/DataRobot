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

dat_raw <- fread('http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests=594d194439b1bc109e02e44d&max_sample_size_only=false')

######################################################
# Organize data
######################################################

dat <- rbind(dat_raw, fill=T)

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

dat[,dataset_bin := cut(dataset_size_GB, c(0, 1.5, 2.5, 5, ceiling(max(dataset_size_GB))), ordered_result=T, include.lowest=T)]
dat[,table(dataset_bin, useNA = 'ifany')]

######################################################
# Organize data
######################################################

dat[,table(Sample_Pct)]
dat[,list(
  RMSE=round(min(`Prediction RMSE`), 2),
  Gini=round(max(`Prediction Gini Norm`), 2)
  ), by='Filename'][order(Gini),]
