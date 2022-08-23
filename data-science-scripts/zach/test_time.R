# Setup
stop()
rm(list=ls(all=T))
gc(reset=T)
library(data.table)
library(ggplot2)
library(ggthemes)

# Load data
keys <- c('job_name', 'module', 'date')
dat <- fread('/Users/zachary/Downloads/Test Time Data - test_modules_data.csv')
dat[,table(date)]
dat[,`mean+sd` := NULL]
setkeyv(dat, keys)

# Do a rolling join to make lags
dat_join <- copy(dat)
dat_join[,date := date + 1]
setkeyv(dat_join, keys)

lags <- setdiff(names(dat), keys)
setnames(dat_join, lags, paste('lag', lags, sep='_'))
dat <- dat_join[dat,,roll=T]
dat <- dat[!is.na(lag_p50),]
rm(dat_join)

# Lookit tests that got worse month over month
# dat[,metric := p25 - lag_p75]
dat[,metric := mean - (lag_mean + lag_stdev)]
dat <- dat[order(metric),]
dat[metric>0, .N]
tail(dat, 1)

# plots
bad_tests <- tail(dat, 1)[, module]
pd <- dat[module %in% bad_tests,]
ggplot(pd, aes(group=date, ymin=min, lower=p25, middle=p50, upper=p75, ymax=max)) + 
  geom_boxplot(stat="identity") + 
  theme_tufte() + facet_wrap(~module)
