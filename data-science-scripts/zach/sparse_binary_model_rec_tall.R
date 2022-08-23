library(data.table)
library(reshape2)
library(rstanarm)
library(lme4)
dat <- fread('~/workspace/data-science-scripts/zach/Binary_data_pipelines.csv')
dat <- melt.data.table(dat, id.vars = 'Pipeline', variable.name='dataset')

dat_test <- dat[is.na(value),]
dat <- dat[!is.na(value),]
setnames(dat, 'value', 'gini')

dat[,Pipeline := factor(Pipeline)]
dat[,dataset  := factor(dataset )]

model <- stan_glmer(
  gini ~ Pipeline + (1 + Pipeline | dataset), data = data.frame(dat),
  algorithm='fullrank',
  sparse=TRUE,
  seed=42
)

summary(model)

pred <- predict(model, newdata=dat_test, se.fit=T)

dat_test[,pred_mean := pred$fit]
dat_test[,pred_sd := pred$se.fit]
dat_test[,pred_1sd := pred_mean - pred_sd]
dat_test[,best := 1:.N == which.max(pred_1sd), by='dataset']
dat_test[best == T,]



model <- stan_glm(
  gini ~ Pipeline + dataset, data=data.frame(dat), 
  algorithm="optimizing",
  QR=FALSE,
  sparse=TRUE,
  seed=42)
summary(model)

pred <- predict(model, newdata=dat_test, se.fit=T)

dat_test[,pred_mean := pred$fit]
dat_test[,pred_sd := pred$se.fit]
dat_test[,pred_1sd := pred_mean - pred_sd]
dat_test[,best := 1:.N == which.max(pred_1sd), by='dataset']
dat_test[best == T,]
