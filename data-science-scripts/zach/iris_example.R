
#Setup
rm(list=ls(all=T))
gc(reset=T)
library(data.table)
library(caret)
library(pbapply)
library(ggplot2)
data(iris)
dat <- data.table(iris)

#Splits
dat <- dat[sample(1:.N),]
train <- dat[1:100,]
test <- dat[101:150,]

train_x <- data.table(model.matrix(Species~0+., train))
test_x <- data.table(model.matrix(Species~0+., test))

train_y <- train[,Species]
test_y <- train[,Species]

#Models
#x=data.table(modelLookup()); x[which(forClass & probModel), sort(unique(model))]
model <- train(
  train_x, train_y, method='lda',
  trControl=trainControl(classProbs = T,  verboseIter=T)
)

#Predict
pred_good = predict(model, test_x, type='prob')
pred_good = rowSums(pred_good^2)
pred_bad = rep(1/length(unique(train_y)), length(pred_good))^2

#Partial dep
pds <- lapply(names(test_x), function(n){
  x = test_x[[n]]
  agg = data.table(
    x=round(x),
    good=pred_good,
    bad=pred_bad
  )
  agg <- agg[,list(
    good=mean(good),
    bad=mean(bad)
    ), by='x']
  agg[,variable := n]
  return(agg)
})
pds = rbindlist(pds)
pds = melt.data.table(pds, id.vars=c('x', 'variable'), variable.name='pd')
ggplot(pds, aes(x=x, y=value, col=pd)) +
  geom_line() + theme_bw() + facet_wrap(~variable,scales='free') +
  theme(legend.position="top")
