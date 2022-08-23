######################################################
# Setup
######################################################

library(data.table)
library(bit64)
library(ggplot2)
library(Hmisc)
library(dplyr)

######################################################
# Download data
######################################################

docker_small <- fread('http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests=59404c19422e2f10da154169&max_sample_size_only=false')
docker_large <- fread('http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests=59404e2b27f1752bee98ad83&max_sample_size_only=false')

######################################################
# Organize data
######################################################

dat <- rbindlist(list(docker_small, docker_large), fill=T)
dat <- dat[!is.na(rows),]
dat <- dat[rows < 400000,]
dat[,summary(rows)]

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
dat[,summary(dataset_size_GB)]

#dat[,table(as.numeric(Sample_Pct))]

######################################################
# Rank ASVM at first autopilot stage
######################################################

dat[,Sample_Size := round(as.integer(Sample_Size))]
dat <- dat[!is.na(Sample_Size),]
dat[,stage := as.integer(factor(Sample_Size)), by=c('pid')]

#Checks
if(FALSE){
  dat[,table(stage)]
  dat[,table(Sample_Pct, stage)]

  dat[Filename == 'yahoo_answers_full_binary_0.95.csv', table(Sample_Pct, Sample_Size)]
  dat[Filename == 'yahoo_answers_full_binary_0.95.csv', table(Sample_Size, stage)]

  dat[Filename == 'Gamblers_80.csv', table(Sample_Pct, Sample_Size)]
  dat[Filename == 'Gamblers_80.csv', table(Sample_Size, stage)]
}

#dat <- dat[stage == 1,]
dat[,table(model_family)]

dat[,is_svm := model_family == 'SVM']

#Ranks
dat[,time_rank := rank(Total_Time_P1_Hours, ties='min'), by='pid']
dat[,gini_rank := rank(1-`Gini Norm_H`, ties='min'), by='pid']
dat[,ram_rank := rank(Max_RAM_GB, ties='min'), by='pid']

#Time Raw
time_order_1 <- dat[,list(mean(Total_Time_P1_Hours)), by='model_family'][order(V1),model_family]
time_order_2 <- dat[stage==4,list(mean(Total_Time_P1_Hours)), by='model_family'][order(V1),model_family]
time_order <- c(time_order_2, setdiff(time_order_1, time_order_2))
ggplot(dat, aes(x=factor(model_family, levels=time_order), y=Total_Time_P1_Hours, col=is_svm)) +
  theme_bw() + geom_boxplot() + facet_wrap(~stage) +
  scale_color_manual(values=c('black', 'red')) + guides(color=FALSE) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  xlab('model_family') +
  ylab('Runtime in Hours') +
  scale_y_log10()

#Time Rank
time_order_1 <- dat[,list(mean(time_rank)), by='model_family'][order(V1),model_family]
time_order_2 <- dat[stage==4,list(mean(time_rank)), by='model_family'][order(V1),model_family]
time_order <- c(time_order_2, setdiff(time_order_1, time_order_2))
ggplot(dat, aes(x=factor(model_family, levels=time_order), y=time_rank, col=is_svm)) +
  theme_bw() + geom_boxplot() + facet_wrap(~stage) +
  scale_color_manual(values=c('black', 'red')) + guides(color=FALSE) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  xlab('model_family') +
  ylab('Runtime Rank (1=fastest, n=slowest)')

#Gini Rank
gini_order <- dat[,list(mean(gini_rank)), by='model_family'][order(V1),model_family]
ggplot(dat, aes(x=factor(model_family, levels=gini_order), y=gini_rank)) +
  theme_bw() + geom_boxplot() + xlab('model_family') + facet_wrap(~stage)

#RAM Rank
ram_order <- dat[,list(mean(ram_rank)), by='model_family'][order(V1),model_family]
ggplot(dat, aes(x=recode_factor(model_family, levels=ram_order), y=ram_rank)) +
  theme_bw() + factor() + xlab('model_family') + facet_wrap(~stage)
