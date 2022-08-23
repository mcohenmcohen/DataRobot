library(data.table)
library(stringi)
library(ggplot2)
library(ggthemes)
dat <- fread('~/workspace/data-science-scripts/zach/CONFIDENTIALCombinedResults.csv')

dat <- dat[,list(
  dataset=sapply(stri_split_fixed(Group, '_', simplify = F), '[', 1),
  partition=sapply(stri_split_fixed(Group, '_', simplify = F), '[', 2),
  target=dataset,
  HoneywellMatLabGP,
  DataRobotGP = `DR GP`,
  DataRobotApp
  )]

ggplot(dat, aes(x=HoneywellMatLabGP, y=DataRobotGP, color=dataset)) + 
  geom_point() + 
  geom_abline(slope=1, intercept = 0) + 
  ggtitle('DataRobot GP vs Honeywell GP') + 
  theme_tufte()

ggplot(dat, aes(x=DataRobotApp, y=DataRobotGP, color=dataset)) + 
  geom_point() + 
  geom_abline(slope=1, intercept = 0) + 
  ggtitle('DataRobot App vs DataRobot GP') + 
  theme_tufte()

dat_tall <- melt.data.table(
  dat, 
  measure.vars=c('HoneywellMatLabGP', 'DataRobotGP', 'DataRobotApp'),
  id.vars=c('dataset', 'partition', 'target')
  )

ggplot(dat_tall, aes(y=value, x=variable)) + 
  geom_boxplot() + 
  theme_tufte()

ggplot(dat_tall, aes(y=value, x=variable)) + 
  geom_violin() + 
  theme_tufte()
