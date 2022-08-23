# https://twitter.com/peterwildeford/status/1545410730630995975
set.seed(42)
N = 100000
pdf = c(.05, .05, .05, .05 ,.05)

# Analytic CDF
cdf = cumsum(pdf)
print(cdf)

# Monte Carlo CDF
x = runif(N*length(pdf))
x = matrix(x, nrow=N, ncol=length(pdf))
x = (x <= 0.05) * 1
mean(matrixStats::rowMaxs(x))  # .25
