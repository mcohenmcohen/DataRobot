library(data.table)
dat = fread('~/Downloads/iip_zengetuhi_2013.csv')
for(var in setdiff(names(dat), 'iip')){
  if(is.numeric(dat[[var]])){
    set(dat, j=var, value=dat[[var]]*100)
  }
}
fwrite(dat, '~/Downloads/iip_zengetuhi_2013_10x_X.csv')

library(data.table)
dat = fread('~/Downloads/iip_zengetuhi_2013.csv')
dat[,iip := iip * 10]
fwrite(dat, '~/Downloads/iip_zengetuhi_2013_10x_y.csv')

library(data.table)
dat = fread('~/Downloads/iip_zengetuhi_2013.csv')
dat[,iip := iip * 10]
fwrite(dat, '~/Downloads/iip_zengetuhi_2013_10x_X_y.csv')