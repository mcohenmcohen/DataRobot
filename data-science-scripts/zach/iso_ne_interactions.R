library(data.table)
library(anytime)
dat <- fread('https://s3.amazonaws.com/datarobot_public_datasets/ISE_NE_8_YEARS_one_date_col.csv')
dat[,d := anytime(date)]
dat[,hour := hour(d)]
dat[,table(hour)]
dat[,day := wday(d)]
dat[,month := month(d)]
dat[,hourXday := paste0('H', hour, 'D', day)]
dat[,hourXmonth := paste0('H', hour, 'M', month)]
dat[,hourXdayXmonth := paste0('H', hour, 'D', day, 'M', month)]
dat[,dayXmonth := paste0('D', day, 'M', month)]
dat[,c('d', 'hour', 'day', 'month') := NULL]
fwrite(dat, '~/datasets/ISO_NE_WITH_MANUAL_INTERACTIONS.csv')
dat[,table(hourXday)]
