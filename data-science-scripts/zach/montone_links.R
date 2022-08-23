set.seed(42)
sigmoid = function(x)
  1 / (1 + exp(-x))
probit = function(x)
  pnorm(x)
cloglog = function(x)
  1-exp(-exp(x))
x <- sort(rnorm(1000))
plot(sigmoid(x) ~ x, type='l', col='black')
lines(probit(x) ~ x, type='l', col='black')
lines(cloglog(x) ~ x, type='l', col='black')


plot(probit(x) ~ x, type='l', col='red')
lines(sigmoid(x*1.6) ~ x, type='l', col='black')

plot(cloglog(x) ~ x, type='l', col='red')
lines(sigmoid(x*1.6+.5) ~ x, type='l', col='black')

