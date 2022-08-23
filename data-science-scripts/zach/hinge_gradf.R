library(Laurae)
library(Deriv)
rmse = function(a, b) sqrt(mean((a-b)^2))

# 1x
fc <- function(x, a=1, k=0){out=x-k; a*ifelse(out > 0, out, 0)}
fc_ref <- function(x) {x}
res = SymbolicLoss(fc = fc,
             fc_ref = fc_ref,
             verbose = TRUE,
             plotting = TRUE)

res$grad
res$hess

mydata <- pmax(-10:10, 3)
opt_me <- function(par) rmse(mydata, par[1]*pmax(mydata-par[2], 0))
opt_me(c(1,0))
optim(c(1, 0), opt_me, gr=function(par){mean(par[1] * ifelse(mydata-par[2]>0,1,0))})

# 2
fc <- function(x, a=1, k=0){out=k-x; a*ifelse(out > 0, out, 0)}
fc_ref <- function(x) {-x}
res = SymbolicLoss(fc = fc,
                   fc_ref = fc_ref,
                   verbose = TRUE,
                   plotting = TRUE)

res$grad
res$hess

