# Copyright (c) 2017 DataRobot
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#Load libraries
library(data.table)
library(caret)
library(caTools)

#Setup
set.seed(42)
nrows <- 2000
ncols <- 500
noise_vars <- 200
pct <- .50

#Simulate
X <- matrix(rnorm(nrows*ncols*3), ncol=ncols)
CF <- runif(ncols, min=-1, max=1)
CF[sample(1:ncols, noise_vars)] <- 0
Y <- X %*% CF + rnorm(nrows, sd=10)

#Turn into classification
cutoff <- quantile(Y, 1-pct)
Y <- as.integer(Y > cutoff)
table(Y) / nrows

#Combine into a single dataset
dat <- data.table(Y=Y, X)

#Split train/train/test
train_a <- dat[1:nrows,]
train_b <- dat[(nrows+1):(nrows*2),]
test <- dat[(nrows*2+1):(nrows*3),]
all(dim(train_a) == dim(test))
all(dim(train_b) == dim(test))

#Fit models
model_a <- glm(Y ~ ., train_a, family='binomial')
model_b <- glm(Y ~ ., train_b, family='binomial')

#Predict on test set
pred_a <- predict(model_a, test, type = 'response')
pred_b <- predict(model_b, test, type = 'response')

#100% mis-match at 6 decimal places
sum(round(pred_a,6) == round(pred_b,6))

#Plot shows little relationship between the 2 models
plot(pred_a ~ pred_b)

#Little agreement between the 2 models
cor(pred_a, pred_b)
summary(abs(pred_a - pred_b))

#Agreement looks a little better on the linear, rather than log scale
#But there is still clearly a big disagreement
pred_a <- predict(model_a, test, type = 'link')
pred_b <- predict(model_b, test, type = 'link')
cor(pred_a, pred_b)
plot(pred_a ~ pred_b)

#AUC of both models is still pretty good
#Even though they disagree a lot!
preds <- cbind(pred_a, pred_b)
colAUC(preds, test$Y)

#Note that one of the 2 models appears to be better, but this is 100% due to random chance
