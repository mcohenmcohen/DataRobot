library(caret)
library(MASS)
library(VGAM)
library(gbm)

#Modified from caret::twoClassSim, see ?twoClassSim
set.seed(1)
n <- 10000
intercept <- 0
sigma <- matrix(c(2, 1.3, 1.3, 2), 2, 2)
X <- data.frame(MASS::mvrnorm(n = n, c(0, 0), sigma))
names(X) <- paste("TwoFactor", 1:2, sep = "")
X <- cbind(X, matrix(runif(n * 3, min = -1), ncol = 3))
colnames(X)[3:5] <- paste("Nonlinear", 1:3, sep = "")
lp <- intercept +
  -4 * X$TwoFactor1 +
  4  * X$TwoFactor2 +
  2  * X$TwoFactor1 * X$TwoFactor2 +
  (X$Nonlinear1^3) +
  2 * exp(-6 * (X$Nonlinear1 - 0.3)^2) +
  2 * sin(pi * X$Nonlinear2 * X$Nonlinear3)

#Convert to a zero-inflated poisson distribution:
lp <- (lp-min(lp)) / (diff(range(lp)) + 0.1)
y <- qzipois(lp, lambda=3, pstr0 = .40)
table(y)

#Fit a GBM
train_idx <- sample(c(T, F), nrow(X), replace=TRUE)
model <- train(
  X[train_idx,],
  y[train_idx],
  method = 'gbm',
  distribution = 'poisson',
  tuneGrid=expand.grid(
    interaction.depth=5,
    n.trees=1:20,
    shrinkage = .1,
    n.minobsinnode = 10
  ),
  trControl = trainControl(method='cv', number=5)
)
min(model$results$RMSE)
plot(model)

#Predict on test set
p <- predict(model, X[!train_idx,], type='raw')
summary(p)
sum(p==0)

