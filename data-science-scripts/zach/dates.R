set.seed(42)
N <- 5000
x <- data.frame(
  a = sample(as.Date(1:floor(N/250), origin='2000-01-01'), N, replace=T),
  b = sample(as.Date(1:floor(N/250), origin='2000-01-01'), N, replace=T)
)

x[['y']] <- as.integer(x[['a']] == x[['b']])
table(x[['y']])
write.csv(x, '~/datasets/dates.csv', row.names=F)

y <- x
y[['a']] <- as.integer(y[['a']]) / 1000
y[['b']] <- as.integer(y[['b']]) / 1000
cor(y)
write.csv(y, '~/datasets/dates_int.csv', row.names=F)
head(x)
