x <- data.frame(
  t = c(rep(1, 4), rep(0, 5000))
)
x$v1 <- runif(nrow(x))
x$v2 <- runif(nrow(x))
x$v3 <- runif(nrow(x))
x$v4 <- runif(nrow(x))
x$v5 <- runif(nrow(x))

write.csv(x, '~/datasets/example.csv')
