# Setup
rm(list = ls(all=T))
gc(reset=T)
library(rvest)
library(quantmod)
library(anytime)
library(data.table)

# Load SPY
getSymbols('SPY', src="alphavantage", from="1900-01-01", auto.assign=TRUE, api.key='B6JUWOCNHEA12M19', output.size='full', adjusted=TRUE)
#head(SPY)
#chartSeries(tail(diff(log(SPY)), 100))

# Preprocess data
dat <- as.data.table(diff(log(SPY)))
setnames(dat, gsub('SPY.', '', names(dat)))
dat[,Close := Adjusted]
dat[,Target := c(Close[2:.N], NA)]
dat <- dat[2:(.N-1),]  # First and last row have NAs
dat[,index := anytime(index)]
#dat[,plot(Adjusted ~ index)]
dat[,index := NULL]
stopifnot(all(sapply(dat, is.finite)))

dat <- dat * 100
fwrite(dat, '~/datasets/SPY.csv')
system('head /Users/zachary/datasets/SPY.csv')
system('tail /Users/zachary/datasets/SPY.csv')
