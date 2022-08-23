# Manual implementation of the model described in:
# https://peerj.com/preprints/3190.pdf

data("AirPassengers")
N <- length(AirPassengers)
x = 1:N
y = as.numeric(AirPassengers)


train_rows <- 1:(N*.80)
test_rows <- (N*.80+1):N
x_train <- x[train_rows]
y_train <- y[train_rows]

x_test <- x[test_rows]
y_test <- y[test_rows]


base_rate <- 1
max_change_points <- 10

s <- rep(0, max_change_points)
d <- rep(0, max_change_points)

# NVM FUCK THIS