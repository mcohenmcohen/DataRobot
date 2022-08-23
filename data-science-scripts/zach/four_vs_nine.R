#https://www.kaggle.com/c/digit-recognizer/data
library(data.table)
library(readr)
set.seed(42)
x = fread('~/Desktop/train.csv')
x = x[label %in% c(4,9),]
x[,label := as.integer(label==9)]
setnames(x, 'label', 'isNine')
rem <- sapply(x, function(x) length(unique(x)))
for(var in names(rem)[rem==1]){
  set(x, j=var, value=NULL)
}
for(var in setdiff(names(x), 'isNine')){
  newx = scale(x[[var]], center = T, scale = T)
  newx = newx + rnorm(length(newx))
  set(x, j=var, value=newx)
}
write_csv(x, '~/datasets/409_noisy.csv')

library(data.table)
library(readr)
x = fread('~/Desktop/train.csv')
x = x[label %in% c(5,3),]
x[,label := as.integer(label==5)]
setnames(x, 'label', 'isFive')
write_csv(x, '~/datasets/five_vs_three.csv')

library(data.table)
library(readr)
x = fread('~/Desktop/train.csv')
x = x[label %in% c(5,8),]
x[,label := as.integer(label==5)]
setnames(x, 'label', 'isFive')
write_csv(x, '~/datasets/five_vs_eight.csv')
