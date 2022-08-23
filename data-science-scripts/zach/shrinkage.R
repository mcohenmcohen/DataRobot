set.seed(42)
x <- sort(runif(1e5, min=-.20, max=.20))

shrink_l1 <- function(x, f){
  shrinkage = abs(x)/(f+abs(x))
  return(x * shrinkage)
}

shrink_l2 <- function(x, f){
  shrinkage = x^2/(f+x^2)
  return(x * shrinkage)
}

plot(1*x ~ x, type='l')
f=.01
lines(shrink_l1(x, f) ~ x, col='blue')
lines(shrink_l2(x, f) ~ x, col='red')
title(paste0('f=', f))
legend(-4, 4, legend=c("l1", "l2"), col=c("blue", "red"), lty=1)
