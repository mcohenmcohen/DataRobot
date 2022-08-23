library(mcomp)
library(forecast)
library(pbapply)
data(M3)

res <- pbsapply(M3, function(x){
  mod <- ets(x$x)
  f <- forecast(mod, length(x$xx))
  return(accuracy(f, x$xx)['Test set','MAPE'])
})

res <- sort(res)
pick <- c(head(res, 5), tail(res, 5))
round(pick, 2)
dev.off()
par(mfrow=c(5,2))
n <- names(pick)
for(i in 1:(length(n)/2)){

  a <- n[[1]]
  b <- n[[length(n)]]

  plot(M3[[a]])
  plot(M3[[b]])

  n <- setdiff(n, c(a,b))
}

n <- 'N2602'
mod <- ets(M3[[n]]$x, restrict=FALSE)
#mod <- auto.arima(M3[[n]]$x, stepwise=FALSE, trace=TRUE)
#mod <- thetaf(M3[[n]]$x)
f <- forecast(mod, length(M3[[n]]$xx))
dev.off()
plot(f)
lines(M3[[n]]$xx, col='red')
