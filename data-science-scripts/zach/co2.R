library(data.table)
library(datarobot)
co2_raw <- fread('ftp://aftp.cmdl.noaa.gov/products/trends/co2/co2_mm_mlo.txt', skip=72)
temp_raw <- fread('http://berkeleyearth.lbl.gov/auto/Global/Complete_TAVG_complete.txt', skip=35)

co2 <- data.table(
  date = seq.Date(from=as.Date('1958-03-01'), length.out=nrow(co2_raw), by='month'),
  co2 = co2_raw[[5]]
)

temp <- data.table(
  date = seq.Date(from=as.Date('1750-01-01'), length.out=nrow(temp_raw), by='month'),
  temp_anomaly = temp_raw[[3]]
)

both <- merge(co2, temp, by='date')
fwrite(both, '~/datasets/co2_temp.csv')

plot(temp_anomaly~date, both, type='l', col='red')
plot(co2~date, both, type='l', col='blue')
plot(temp_anomaly~co2, both)
