wrap_deviance <- function(metric){
 out <- function(actual, pred, weight=NULL){
   if(is.null(weight)){
     weight <- rep(1, length(pred))
   }
   mean(metric()$dev.resids(actual, pred, weight))
 }
 return(out)
}

poisson_deviance <- wrap_deviance(poisson)
binomial_deviance <- wrap_deviance(binomial)
gaussian_deviance <- wrap_deviance(gaussian)

set.seed(42)
pred <- runif(10)
act <- sample(0:1, length(pred), replace=T)

poisson_deviance(act, pred)
binomial_deviance(act, pred)
gaussian_deviance(act, pred)
