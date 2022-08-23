rm(list=ls(all=T))
gc(reset=T)
set.seed(42)
N <- 1000
a <- runif(N)
b <- sample(letters, N, replace=T)
b[sample(1:N, N/10)] <- 'Infinity'
b[sample(1:N, N/10)] <- '-Infinity'
dat <- data.frame(a, b)
X <- model.matrix(~., dat)

cf <- rnorm(ncol(X))
noise <- runif(N)
y <- X %*% cf + noise

a[sample(1:N, N/10)] <- 'Infinity'
a[sample(1:N, N/10)] <- '-Infinity'

dat <- data.frame(y, a, b)
write.csv(dat, '~/datasets/Infinity.csv', row.names=F)
