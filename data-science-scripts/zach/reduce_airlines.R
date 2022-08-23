library(data.table)
names(fread('~/datasets/airlines_10gb.csv', nrows=10))
dat <- fread(
  '~/datasets/airlines_10gb.csv',
  select=c(
    'Year',
    'Month', #Make categorical
    'DayofMonth', #REMOVE, more data
    'DayOfWeek', #Make categorical
    'DepTime',
    'UniqueCarrier',
    'FlightNum',
    'TailNum', #REMOVE, more data
    'Origin',
    'Dest',
    'Distance',
    'ArrDelay'
  ), nrows=20000000)
dat <- dat[!is.na(ArrDelay),]

fname <- '~/datasets/airlines_reduced.csv'
write.csv(dat, fname, row.names=F)
gdata::humanReadable(file.size(fname))

mm <- dat[is.finite(Distance),list(
  dist=mean(Distance, na.rm=TRUE),
  n=length(unique(Distance))), by=c('Origin', 'Dest')]
summary(mm)
