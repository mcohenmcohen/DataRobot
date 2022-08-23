# https://math.stackexchange.com/questions/65641/i-need-to-define-a-family-one-parameter-of-monotonic-curves
set.seed(42)
N = 1000
x=sort(runif(N))
sigmoid = function(x) 1 / (1 + exp(-x))
probit = function(x) pnorm(x)
cloglog = function(x) 1-exp(-exp(x))
logit = function(x) log(x) - log(1-x)
expit = function(x) 1/(1+exp(-x))
plot((1-x**0.25)**1/.25 ~ x, type='l')
lines((1-x**1.75)**1/1.75 ~ x, type='l')
lines(logit(x) ~ x, type='l')


# Generalize logistic
# https://en.wikipedia.org/wiki/Generalised_logistic_function
# Y(t)=A+{K-A \over (C+Qe^{{-Bt}})^{{1/\nu }}}

set.seed(42)
N = 10000
x = sort(rnorm(N))
gen_logistic = function(x, v=1, Q=1){
  1 / (1 + Q * exp(-1*x)) ^ (1/v)
}
plot(gen_logistic(x) ~ x, type='l', col='red')
for (Q in c(.1, .5, 1, 2, 5)) {
  for (v in c(.5, 1, 2)) {
    lines(gen_logistic(x, Q=Q, v=v) ~ x)
  }
}
lines(gen_logistic(x*1.6) ~ x)
lines(gen_logistic(x*1.6+.5) ~ x)
lines(sigmoid(x) ~ x, col='red')
lines(probit(x) ~ x, col='blue')
lines(cloglog(x) ~ x, col='green')
    
