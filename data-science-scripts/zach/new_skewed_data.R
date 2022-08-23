set.seed(42)
N <- 1000
normal_dist <- rnorm(N)
skewed_dist <- rgamma(N, .1)

hist(normal_dist)
hist(skewed_dist)

print(paste('Normal Distribution mean + 20%: ', round(mean(normal_dist) * 1.20, 2)))
print(paste('Normal Distribution 70th percentile: ', round(quantile(normal_dist, .70), 2)))

print(paste('Skewed Distribution mean + 20%: ', round(mean(skewed_dist) * 1.20, 2)))
print(paste('Skewed Distribution 70th percentile: ', round(quantile(skewed_dist, .70), 2)))