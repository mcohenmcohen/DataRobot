# Copyright (c) 2016 DataRobot
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

#Value curve function
value_curve <- function(
  pred, # Predicted.  Must be a number between 0 and 1
  act, # Actual  Must be an integer between 0 and 1
  tp=1, # Value of a true positive (negative = value)
  fp=-1, # Value of a false positive (negative = value)
  tn=1, # Value of a true negative (negative = value)
  fn=-1, # Value of a false negative (negative = value)
  maximize=TRUE, # TRUE to maximize value.  Set to false if your inputs are values, not values
  plotit=TRUE, # Whether or not to plot the value curve
  return_full_curve=FALSE # If TRUE return the full curve, if FALSE, return the best point
){

  #Checks
  library(ROCR)

  stopifnot(is.numeric(pred))
  stopifnot(min(round(pred, 8)) >= 0.0)
  stopifnot(max(round(pred, 8)) <= 1.0)

  stopifnot(is.numeric(act))
  stopifnot(min(act) == 0)
  stopifnot(max(act) == 1)
  stopifnot(length(unique(act)) == 2)

  stopifnot(length(act) == length(pred))

  #Construct curve
  N <- length(act)
  curve <- ROCR::prediction(pred, act)
  threshold <- curve@cutoffs[[1]]
  metrics <- cbind(
    tp = curve@tp[[1]],
    fp = curve@fp[[1]],
    tn = curve@tn[[1]],
    fn = curve@fn[[1]]
  )

  #Calculate the value curve
  value_matrix <- rbind(tp, fp, tn, fn)
  value <- metrics %*% value_matrix
  curve <- data.frame(threshold, value)
  curve$threshold[curve$threshold > max(pred)] <- max(pred) * 1.10
  curve$threshold[curve$threshold < min(pred)] <- min(pred) * 0.90

  #Optimize value

  if(maximize){
    best <- which.max(curve$value)
  } else{
    best <- which.min(curve$value)
  }
  curve$best <- 0L
  curve$best[best] <- 1L

  #Plot
  if(plotit) {
    plot(value ~ threshold, curve, type='l')
    points(value ~ threshold, curve, col='black', pch=19, cex=0.5)
    points(value ~ threshold, curve[best,], col='red', pch=19, cex=2)
  }

  #Return data
  if(return_full_curve){
    reutrn(curve)
  }else{
    return(curve[best,])
  }
}

#Example
library(caret)
library(C50)
library(glmnet)
data(churn)
set.seed(42)
model <- train(
  churn ~ ., churnTrain, method='glmnet', metric='ROC',
  tuneGrid = expand.grid(alpha=0:1, lambda=seq(0, 0.10, length=25)),
  trControl=trainControl(
    method='cv', number=10,
    verboseIter = TRUE,
    classProbs = TRUE, summaryFunction = twoClassSummary
  ))
p <- predict(model, churnTest, type='prob')$yes
a <- as.integer(churnTest$churn == 'yes')

#We've devised an intervention for churners: a $100 bill credit.
#We've found that if we offer this credit to churners, they on average stay with us long enough to earn us an additional $110.
#If we offer this credit to non-churners, they increase their usage a little bit, at earn us $99 more on average.
#Therefore, the treatment increases value by $10 for churners (110 - 100 = 10)
#And decreases value by $1 for non-churners (99 - 100 = -1)
#Lets say that true negatives and false negatives are worth nothing, since we don't intervene in these cases and nothing changes:
value_curve(p, a, tp=10, fp=-1, tn=0, fn=0)

#According to our value curve, if we use a threshold of 0.1389597 for our churn model (or about 14% chance of churn) to decide when to intervene, we expect to make an additional $1560
