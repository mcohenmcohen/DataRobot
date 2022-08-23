library(VGAM)  # for expit

bad_logit = function(x, eps=1e-5){
  x_t = pmin(pmax(x, eps), 1 - eps)  # make sure within (0,1)
  return(log(x_t - log(1 - x_t)))
}

good_logit = function(x, eps=1e-5){
  x_t = pmin(pmax(x, eps), 1 - eps)  # make sure within (0,1)
  return(log(x_t) - log(1 - x_t))
}

expit = function(x){
  exp(x) / (1+exp(x))
}

set.seed(42)
x <- sort(runif(1000))

# Plot 1
plot(good_logit(x) ~ x, col='black', type='l')
lines(bad_logit(x) ~ x, col='red', type='l')
legend(
  0.10, 5, 
  legend=c("Good Logit", "Bad Logit"),
  col=c("black", "red"), 
  lty=1, cex=0.8)

# Plot 2
plot(expit(good_logit(x)) ~ x, col='black', type='l')
lines(expit(bad_logit(x)) ~ x, col='red', type='l')
legend(
  0.10, 5, 
  legend=c("Good Logit", "Bad Logit"),
  col=c("black", "red"), 
  lty=1, cex=0.8)

# Plot 3
plot(good_logit(x) - bad_logit(x) ~ x, col='black', type='l')

# Plot 4
plot(good_logit(x) ~ x, col='black', type='l')
lines(bad_logit(x) ~ x, col='red', type='l')
lines(cloglog(x) ~ x, col='blue', type='l')
lines(probit(x) ~ x, col='green', type='l')
legend(
  0.10, 5, 
  legend=c("Good Logit", "Bad Logit", "cloglog", "probit"),
  col=c("black", "red", "blue", "green"), 
  lty=1, cex=0.8)



