library(data.table)
library(reshape2)
library(ggplot2)
dat_full <- fread('http://shrink-mongo.dev.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests=5857a6ab3e4ffa11a8a8ca37&max_sample_size_only=false')

dat <- copy(dat_full)
setnames(dat, make.names(names(dat)))

dat <- dat_full[Sample_Pct == 16 | Sample_Pct == 64,]
dat[,rank := rank(1-Gini.Norm_H, ties.method= "random"), by=c('pid', 'Sample_Pct')]

dat <- dat[,list(
  Filename,
  reference_model,
  pid,
  Blueprint,
  tasks_args,
  main_task,
  Sample_Pct,
  rank
)]
dat <- melt.data.table(dat, measure.vars = c('rank'))
dat <- dcast.data.table(dat, Filename + pid + Blueprint + tasks_args + reference_model ~ variable + Sample_Pct)
dat <- dat[!is.na(rank_16) & !is.na(rank_64),]

ggplot(dat, aes(x=factor(rank_16), y=rank_64)) + geom_boxplot() + theme_bw()
ggplot(dat, aes(x=factor(rank_64), y=rank_16)) + geom_boxplot() + theme_bw()

summary(lm(rank_64 ~ rank_16, dat))

dat[rank_16 == 2,]
a <- dat_full[Filename == 'BostonPublicRaises_80.csv' & Blueprint == "{u'1': [[u'CAT'], [u'PDM3 cm=50000;dtype=float32;sc=10'], u'T'], u'2': [[u'1'], [u'ENETCD a=0;t_n=5;t_a=2;t_m=RMSE'], u'P']}",]
b <- sapply(a, function(x) length(unique(x))) > 1
a[,b,with=F]
