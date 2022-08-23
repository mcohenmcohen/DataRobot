library(gbm)
set.seed(42)
x = rgamma(10000,.1, scale=200)
summary(x)
noise = rgamma(10000,.1, scale=200)
y = exp(1.5*log(x))/100 + 5 * noise
summary(y)
hist(y)
model = gbm(log1p(y) ~ x, distribution='gaussian', verbose=T, shrinkage=0.1, n.trees=10000)
p = expm1(predict(model, n.trees=10000))
max(p)
max(y)
error = p-y
summary(error)
