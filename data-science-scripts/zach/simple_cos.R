set.seed(42)
N = 50
a = rnorm(N)
b = rnorm(N)
y = 2 * cos(a) + 3 * b^2 + rnorm(N)
out = data.frame(y, a, b)
write.csv(out, '~/datasets/simple_cos.csv', row.names=F)
