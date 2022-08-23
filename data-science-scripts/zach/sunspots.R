library(data.table)
library(anytime)
dat = fread('https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv')
dat[,Month := anytime(Month)]
dat[,Month := format(Month, '%Y-%m-%d')]
fwrite(dat, '~/Downloads/sunspots.csv')

data(co2)
year = as.integer(floor(index(co2)))
month = as.integer((index(co2) %% 1) * 12 + 1)
out = data.table(
  date=as.Date(ISOdate(year, month, 1)),
  target = as.numeric(co2)
)
fwrite(out, '~/Downloads/co2.csv')
