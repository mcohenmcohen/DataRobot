library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
setwd('~/workspace/data-science-scripts/zach/')
#http://discourse.mc-stan.org/t/approximate-gps-with-spectral-stuff/1041/32
#http://proceedings.mlr.press/v28/le13-supp.pdf
#https://bitbucket.org/flaxter/random-fourier-features-in-stan/src/8d303d3d36d6?at=master
#https://github.com/bbbales2/basketball/blob/master/models/rff_bernoulli.stan
N = 20000
x = runif(N,0,1)
x = x[order(x)]
y = sin(1*pi*x) + cos(2*pi*x) + cos(3*pi*x) + rnorm(N,0,.1)

x = as.numeric(scale(x, center=T, scale=T))
y = as.numeric(scale(y, center=T, scale=T))

k=10
data = list(y=y,x=x,n=length(y),k=k,bw=8,omega=rnorm(k))

plot(y~x)

# fixed bandwidth
m1 = stan_model("rff1.stan")
fit = sampling(m1, data=data, iter=100, warmup=100, chains=16)
out = extract(fit)

plot(x,y,pch=18,col="gray",main="Fixed lengthscale")
lines(x,colMeans(out$fhat),col="blue")

# learn the bandwidth
m2 = stan_model("rff2.stan")
fit = sampling(m2, data=data, iter=200, warmup=100, chains=8)
out = extract(fit)
print(fit,c("bw","lp__"))

plot(x,y,pch=18,col="gray",main="Learnt lengthscale")
lines(x,colMeans(out$fhat),col="red")
