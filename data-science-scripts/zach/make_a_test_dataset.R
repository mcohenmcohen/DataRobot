set.seed(42)
library(data.table)
library(readr)
dat = fread('https://s3.amazonaws.com/datarobot_public_datasets/bike_sharing.csv')
dat[,a := runif(.N)]
dat[,b := runif(.N)]
dat[,c := runif(.N)]
dat[,d := runif(.N)]
dat[,e := runif(.N)]
setnames(dat, 'a', '日 (Lag1)')
setnames(dat, 'b', '日 [Lag1]')
setnames(dat, 'c', '(Lag1) 日')
setnames(dat, 'd', '[Lag1] 日')
setnames(dat, 'Weekday', '平日')
setnames(dat, 'Season', 'シーズン')
setnames(dat, 'y', 'ターゲット')
setnames(dat, 'date', '日付')
write_csv(dat, '~/datasets/fear_test')
