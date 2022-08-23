
set.seed(42)
N <- 1000
a <- c(rep(1, N/2), rep(0, N/2))
b <- c(rep(0, N/2), rep(1, N/2))
y <- a * 1 + b * 10 + rnorm(N, mean=0, sd=1)

dat_1 <- data.frame(y, a, b)
model_1 <- glm(y ~ a + b, data=dat_1)
summary(model_1)
summary(dat_1)

dat_2 <- head(dat_1, N*.60)
model_2 <- glm(y ~ a + b, data=dat_2)
summary(model_2)
summary(dat_2)
