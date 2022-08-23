library(data.table)
library(ggplot2)
library(ggthemes)

dat <- fread('~/Downloads/Blended dataset_5e978d2dca2c1313d2305964.csv')
top_features <- fread('~/Downloads/Feature Impact for Light Gradient Boosted Trees Classifier with Early Stopping using 67908 rows.csv')

# Make factors
dat[,DD_IN_30_DAYS := factor(DD_IN_30_DAYS)]
dat[,DD_IN_60_DAYS := factor(DD_IN_60_DAYS)]
dat[,DD_IN_90_DAYS := factor(DD_IN_90_DAYS)]
dat[,DD_IN_180_DAYS := factor(DD_IN_180_DAYS)]

# Tables
dat[,table(DD_IN_30_DAYS, is.na(TIMES_VIEWED_PAYROLL_DD))]
dat[,table(DD_IN_60_DAYS, is.na(TIMES_VIEWED_PAYROLL_DD))]
dat[,table(DD_IN_90_DAYS, is.na(TIMES_VIEWED_PAYROLL_DD))]
dat[,table(DD_IN_180_DAYS, is.na(TIMES_VIEWED_PAYROLL_DD))]

dat[,table(DD_IN_60_DAYS, is.na(TIMES_VIEWED_PAYROLL_DD))]
dat[,table(DD_IN_60_DAYS, is.na(DAYS_SCREEN_ACTIVE))]

# Stats
dat[,list(
  na = sum(is.na(TIMES_VIEWED_PAYROLL_DD)),
  min = min(TIMES_VIEWED_PAYROLL_DD, na.rm=T),
  mean = mean(TIMES_VIEWED_PAYROLL_DD, na.rm=T),
  median = as.numeric(median(TIMES_VIEWED_PAYROLL_DD, na.rm=T)),
  max = max(TIMES_VIEWED_PAYROLL_DD, na.rm=T),
  sd = sd(TIMES_VIEWED_PAYROLL_DD, na.rm=T)
), by='DD_IN_60_DAYS']

dat[TIMES_VIEWED_PAYROLL_DD>93,.N]

# Plots
ggplot(dat, aes(y=log1p(TIMES_VIEWED_PAYROLL_DD), col=DD_IN_60_DAYS)) + geom_boxplot() + theme_tufte()
