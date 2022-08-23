rm(list=ls(all=T))
gc(reset=T)
library(data.table)
x=fread('~/datasets/ISE_NE_8_YEARS.csv', colClasses = c("character", "character", "numeric"))
setnames(x, 'Hour Ending', 'HE')
setnames(x, 'Total Load', 'y')
x[,Date := as.Date(Date)]
x[,date := ISOdatetime(year(Date), month(Date), mday(Date), as.numeric(HE), 0, 0, tz = "America/New_York")]
x[,dst_spring := as.integer(sum(HE=='3') == 0 & month(Date) == 3), by='Date']
x[,dst_fall := as.integer(max(HE=='02X')), by='Date']
x[dst_spring==1,unique(Date)]
#x[is.na(date) & HE == '02X', date := ISOdatetime(year(Date), month(Date), mday(Date), 2, 0, 0, tz = "America/New_York")] # WRONG!
#x[is.na(date) & HE == '2', date := ISOdatetime(year(Date), month(Date), mday(Date), 3, 0, 0, tz = "America/New_York")]
x[dst_spring==1, date := ISOdatetime(year(Date), month(Date), mday(Date), 0, 0, 0, tz = "America/New_York") + 1:23*3600]
x[dst_fall==1, date := ISOdatetime(year(Date), month(Date), mday(Date), 0, 0, 0, tz = "America/New_York") + 1:25*3600]
x[,date := format(date, usetz=FALSE, tz='UTC')]
x[Date == as.Date('2016-03-13'),]
x[Date == as.Date('2016-11-06'),]
x <- x[,list(date, y)]
fwrite(x, '~/datasets/cfds_hourly/ISE_NE_8_YEARS_one_date_col.csv')
stopifnot(!anyNA(x))
system('head -n48 ~/datasets/cfds_hourly/ISE_NE_8_YEARS_one_date_col.csv')
