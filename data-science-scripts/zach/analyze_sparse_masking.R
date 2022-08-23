rm(list=ls(all=T))
gc(reset=T)
library(data.table)
library(ggplot2)
library(ggthemes)
library(scales)

dat <- fread('~/workspace/data-science-scripts/zach/results.csv')
dat[,size := rows * columns * sparsity]
summary(dat)

dat <- melt.data.table(
  dat, 
  measure.vars=c('zach_time', 'josh_time', 'thomas_time'),
  value.name='time',
  variable.name='method'
  )
dat[,method := gsub('_time', '', method, fixed=T)]
summary(lm(time ~ method + size, dat))

ggplot(dat, aes(x=size, y=time, col=method)) + 
  geom_point() + geom_smooth() + 
  theme_tufte()








ggplot(dat, aes(x=sparsity, y=time, col=method)) + 
  geom_point() + geom_smooth() + 
  guide_legend(title.position = 'top') + 
  theme_tufte()

ggplot(dat, aes(x=rows, y=time, col=method)) + 
  geom_point() + geom_smooth() + 
  guide_legend(title.position = 'top') + 
  theme_tufte()

ggplot(dat, aes(x=columns, y=time, col=method)) + 
  geom_point() + geom_smooth() + 
  guide_legend(title.position = 'top') + 
  theme_tufte()
