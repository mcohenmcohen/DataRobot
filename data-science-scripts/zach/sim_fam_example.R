# Setup
rm(list=ls(all=T))
gc(reset=T)
library(data.table)
library(ggplot2)
library(ggthemes)
set.seed(42)

# Simulate data
N_train <- 2e7
N_valid <- 1e4
N_holdout <- 1e5
print(N_train/(N_train + N_valid + N_holdout))
stopifnot(N_train <  0.999 * (N_train + N_valid + N_holdout))

a_train <- rnorm(N_train + N_valid)
b_train <- a_train + rnorm(N_train + N_valid)*.05

a_hold <- rnorm(N_holdout)
b_hold <- rnorm(N_holdout)

dat_train <- data.table(
  set=c(rep('t', N_train), rep('v', N_valid)),
  a=a_train,
  b=b_train
)
dat_test <- data.table(
  set='h',
  a=a_hold,
  b=b_hold)
dat <- rbind(dat_train, dat_test, fill=T, use.names=T)
dat[,target := a - b + rnorm(length(a)) * .1]
dat[,table(set)]

dat[,cor(a, b)]
dat[,cor(a, b), by='set']
dat[,cor(target, a), by='set']
dat[,cor(target, b), by='set']
dat[,cor(target, a-b), by='set']
dat[,as.list(coef(lm(target ~ 0 + a + b))), by='set']
dat[,as.list(coef(lm(target ~ 0 + a + b)))]

dat[,order := runif(.N)]
setorderv(dat, 'order')
dat[,order := NULL]
fwrite(dat, '~/Downloads/fam_vs_redundant.csv')

# Plots
dev.off()
gc(reset=T)

plot_dat <- rbind(
  dat[set %in% c('v', 'h'),],
  dat[sample(which(set=='t'), N_valid),]
)
plot_dat[,table(set)]

if(nrow(plot_dat) <= 10000){
  print(
    ggplot(plot_dat, aes(x=a, y=b, color=set)) +
      geom_point() +
      facet_wrap(~set) +
      scale_color_brewer(palette = "Set1") +
      coord_cartesian() +
      theme_tufte()
  )

  print(
    ggplot(plot_dat, aes(x=a, y=target, color=set)) +
      geom_point() +
      facet_wrap(~set) +
      scale_color_brewer(palette = "Set1") +
      coord_cartesian() +
      theme_tufte()
    )

  print(
    ggplot(plot_dat, aes(x=b, y=target, color=set)) +
      geom_point() +
      facet_wrap(~set) +
      scale_color_brewer(palette = "Set1") +
      coord_cartesian() +
      theme_tufte()
    )

  print(
    ggplot(plot_dat, aes(x=(a-b), y=target, color=set)) +
      geom_point() +
      facet_wrap(~set) +
      scale_color_brewer(palette = "Set1") +
      coord_cartesian() +
      theme_tufte()
    )
}
