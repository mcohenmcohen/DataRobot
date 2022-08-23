library(data.table)
library(acepack)
library(ggplot2)
library(ggthemes)
cls = c(
  "integer", "integer", "character", "character", "character",
  "character", "character", "character", "character", "character",
  "integer", "integer", "integer", "character", "character", "character",
  "character", "integer", "character", "integer", "character",
  "character", "character", "character", "character", "character",
  "numeric", "integer", "integer", "integer", "character", "character",
  "character", "character")
dat = fread('~/datasets/customer/Work_orders_for DR v3.csv', colClasses = cls)
dat[,`GesKosten Ist` := as.numeric(gsub(",", "", `GesKosten Ist`))]
dat[,sum(is.na(`GesKosten Ist`))]
dat = dat[!is.na(`GesKosten Ist`),]
y = dat[['GesKosten Ist']]
x = dat[['GesStunden Ist']]
cor(y, x)
ace(x, y)[['rsq']]
ggplot(dat, aes(x=`GesStunden Ist`, y=`GesKosten Ist`)) +
  geom_point() + geom_smooth() + theme_tufte()

out = dat[`GesStunden Ist` < 400,]
fwrite(out, '~/datasets/customer/work_order_no_outlier.csv')
