library(data.table)
N = 1000
x = seq(-1, 1, length=N)
y =  sign(x) * x**3 - abs(.5*x)
plot(y~x)
rmse <- function(a, b) sqrt(mean((a-b)^2))
fwrite(data.table(y, x), '~/datasets/funny_function.csv')

train_idx <- sort(sample(1:N, N*.80))
test_idx  <- sort(setdiff(1:N, train_idx))

x_train <- x[train_idx]
y_train <- y[train_idx]

x_test <- x[test_idx]
y_test <- y[test_idx]

model <- glm(y_train ~ x_train)
summary(model)

lines(predict(model, newx=x_train)~x_train)

print(rmse(predict(model, newx=x_train), y_train))
print(rmse(predict(model, newx=x_test), y_test))
