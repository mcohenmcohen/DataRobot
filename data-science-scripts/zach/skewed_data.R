library(boot)
set.seed(1)
rows <- 1e4
cols <- 50

X <- matrix(rnorm(rows*cols), ncol=cols)
cf <- rnorm(cols)
cf[-(1:10)] <- 0

Y <- X %*% cf + rnorm(rows) * 5
Y <- inv.logit(Y[,1] / 5)
hist(Y)
Y <- as.integer(Y > .95)
table(Y)
table(Y) / rows

dat <- data.frame(Y, X)
head(dat)

fname <- '~/datasets/extreme_skewed_small.csv'
write.csv(dat, file=fname)
gdata::humanReadable(file.size(fname))
