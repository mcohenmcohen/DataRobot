library(data.table)
library(pbapply)
library(forecast)
dat <- fread('~/datasets/muni.csv')
dat[,PX_Last := strptime(PX_Last, '%m/%d/%y')]
setkeyv(dat, 'PX_Last')
setnames(dat, make.names(names(dat)))
dat[, year := year(PX_Last)]

d#at <- dat[year > 2010,]

par(mfrow=c(3,1))
plot(MUNSMT01.Index ~ PX_Last, dat, type='l')
plot(MUNSMT05.Index ~ PX_Last, dat, type='l')
plot(MUNSMT10.Index ~ PX_Last, dat, type='l')

vars <-  c('MUNSMT01.Index', 'MUNSMT05.Index', 'MUNSMT10.Index')
ets_models <- pblapply(vars, function(v){
  model <- ets(dat[year < 2017,][[v]], ic='bic')
  fcast <- forecast(model, dat[year >= 2017,.N])
  acc <- accuracy(fcast, dat[year >= 2017,][[v]])
  return(
    list(
      model = model,
      fcast = fcast,
      acc = acc
    )
  )
})

ar_models <- pblapply(vars, function(v){
  model <- auto.arima(dat[year < 2017,][[v]], ic='bic', stepwise=F, d=1)
  fcast <- forecast(model, dat[year >= 2017,.N])
  acc <- accuracy(fcast, dat[year >= 2017,][[v]])
  return(
    list(
      model = model,
      fcast = fcast,
      acc = acc
    )
  )
})

lapply(ets_models, '[[', 'acc')
lapply(ar_models, '[[', 'acc')[[3]]

lapply(ar_models, '[[', 'model')
plot(lapply(ar_models, '[[', 'fcast')[[3]])
