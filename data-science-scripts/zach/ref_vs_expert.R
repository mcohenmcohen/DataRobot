#Load Data
library('shrinkR')
library('data.table')
library('reshape2')
library('ggplot2')
library('ggrepel')
library('scales')
library('jsonlite')
library('pbapply')
library('stringi')
library('readr')

#Load base MBtest and new run
#Sys.sleep(43200)
suppressWarnings(dat <- getLeaderboard('5812156ffc5c7411755c04f8'))
setnames(dat, make.names(names(dat), unique=TRUE))
dat[,length(unique(Filename))]

#Calcs
res <- dat[,list(rank = rank(1-Gini.Norm_H), N=.N, reference_model), by=c('Filename')]
res[reference_model == TRUE,summary(rank)]

res <- res[reference_model == TRUE,list(
  rank=min(rank),
  N=N[1]
  ), by='Filename']

#Calcs
res <- dat[,list(gini_norm = max(Gini.Norm_H)), by=c('Filename', 'reference_model')]
res[,reference_model := ifelse(reference_model, 'ref', 'exp')]
res <- dcast.data.table(res, Filename ~ reference_model, value.var='gini_norm')
res[,exp := (1-exp)]
res[,ref := (1-ref)]
res[, gini_improvement := 1 - exp / ref]
res[!is.na(gini_improvement),summary(gini_improvement)]
res[!is.na(gini_improvement),hist(gini_improvement)]
res[,imp := ifelse(exp <= ref, 1, 0)]
summary(res[imp==0,])

#Save for dan
out <- dat[,list(
  Filename,
  Y_Type,
  Sample_Pct,
  main_task,
  reference_model,
  quickrun_model,
  is_blender,
  is_prime,
  Max_RAM,
  holdout_size,
  holdout_scoring_time,
  Total_Time_P1,
  Cached_Time_P1,
  metric,
  RMSE_H,
  Poisson.Deviance_H,
  MAD_H,
  LogLoss_H,
  Gini.Norm_H,
  Gamma.Deviance_H
)]
write_csv(out, '~/datasets/metadatafordan.csv')
