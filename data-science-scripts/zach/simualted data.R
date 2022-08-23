d <- fread('~/datasets/downSampled_rollup_signals_1day_farWindow_fg5.csv')
d <- d[order(runif(.N)),]
d[,n := 0L]
d[isEvent == 'noEvent',n := 1:.N]
cnt <- d[,sum(isEvent=='event')]
d <- d[n <= cnt * 750,]
d[,n := NULL]
table(d$isEvent)
d[,sum(isEvent=='noEvent') / sum(isEvent=='event')]
write.csv(d, '~/datasets/customer_data_500.csv', row.names=FALSE)


ratio <- 975
n <- 72
b <- ratio * n
dat <- data.table(
  id = 1:(n + b)
)
dat[,target := 0L]
dat[id <= n,target := 1L]
dat[,x := round(runif(.N), 1)]
dat[,sum(target==0) / sum(target==1)]
write.csv(dat, '~/datasets/dat_n_to_1.csv', row.names=FALSE)
