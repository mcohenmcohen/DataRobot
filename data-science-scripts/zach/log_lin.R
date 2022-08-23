set.seed(42)
N <- 1000
x <- seq(1, 100, length=N)
y <- x
y_log <- log(y)

model_lin <- lm(y~x)
model_log <- lm(y_log~x)

pred_lin <- predict(model_lin)
pred_log <- exp(predict(model_log))

error <- rnorm(N)
pred_lin <- predict(model_lin) + error
pred_log <- exp(predict(model_log) + error)

plot(pred_lin ~ y, col='black')
plot(pred_log ~ y, col='black')
