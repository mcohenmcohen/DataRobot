
x = c(32, 64, 80, 100)
y = c(22.90, 45.29, NA, NA)
dat = data.frame(y, x)
model = lm(y ~ x, data=dat)
summary(model)
predict(model, newdata=dat)
