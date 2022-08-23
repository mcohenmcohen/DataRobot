set.seed(42)
logit = function(x)
  1 / (1 + exp(-x))
cloglog = function(x)
  1-exp(-exp(x))
probit = function(x)
  pnorm(x)
x <- sort(rnorm(1000))
plot(logit(x) ~ x, type='l', col='black')
lines(probit(x) ~ x, type='l', col='red')
lines(cloglog(x) ~ x, type='l', col='blue')


# http://m-hikari.com/ams/ams-2014/ams-85-88-2014/epureAMS85-88-2014.pdf
approx_probit1 = function(x)
  2^(-22^(1-41^(x/10)))

# http://www.jiem.org/index.php/jiem/article/viewFile/60/27
approx_probit2 = function(x)
  1/(1+exp(-(0.07056*x^3+1.5976*x)))

# http://www.hrpub.org/download/20140305/MS7-13401470.pdf
approx_probit3 = function(x){
  a = 0.147
  term1 = (x^2)/2
  term2 = 4/pi + a*term1
  term3 = 1 + a*term1
  out = -term1 * term2 / term3
  out = 1 - exp(out)
  out = out ^ 0.5
  out = (1+out)/2
  return(out)
}
approx_probit4 = function(x){
  out = 0.5 + 0.5 * sqrt(1 - exp(-sqrt(pi/8* x^2)))
  return(out)
}
approx_probit5 = function(x){
  out = 1 + exp(-pi*x/sqrt(3))-1
  return(out)
}

x <- sort(rnorm(1000))
plot(probit(x) ~ x, type='l', col='black')
plot(approx_probit5(x) ~ x, type='l', col='red')

max(abs(probit(x)-approx_probit1(x)))
max(abs(probit(x)-approx_probit2(x)))
max(abs(probit(x)-approx_probit5(x)))



lines(logit(x) ~ x, col='blue')
lines(cloglog(x) ~ x, col='blue')