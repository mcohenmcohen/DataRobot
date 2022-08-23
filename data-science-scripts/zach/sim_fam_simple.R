# Setup
rm(list=ls(all=T))
gc(reset=T)
library(data.table)
library(ggplot2)
library(ggthemes)
set.seed(42)

# Simulate data
N <- 1e4
a <- rnorm(N)
b <- a + rnorm(N)*.05
c <- a + rnorm(N)*.05

dat <- data.table(a, b, c)
dat[,target := a - b + rnorm(length(a)) * .01]
fwrite(dat, '~/Downloads/fam_vs_red_simple.csv')

# Lookit data
dat[,cor(a, b)]
dat[,cor(target, a)]
dat[,cor(target, b)]
dat[,cor(target, c)]
dat[,cor(target, a-b)]
dat[,as.list(coef(lm(target ~ 0 + a + b + c)))]
dat[,order := runif(.N)]
setorderv(dat, 'order')
dat[,order := NULL]
dim(dat)
