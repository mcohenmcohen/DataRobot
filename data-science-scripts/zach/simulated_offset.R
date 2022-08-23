library(data.table)
set.seed(42)
nrows = 1000
ncols = 10
n = nrows * ncols
X = matrix(rnorm(n), ncol=ncols)
CF = rnorm(ncols)
offset = runif(nrows)
noise = rnorm(nrows) * 2
y = offset + X %*% CF + noise
dat = data.table(
  y=y[,1],
  offset=offset,
  X
)
fwrite(dat, '~/datasets/simulated_offset.csv')


set.seed(42)
x = rnorm(n)
y = cumsum(x)
plot(y, type='l')


