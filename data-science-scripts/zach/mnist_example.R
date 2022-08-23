
#Setup
rm(list=ls(all=T))
gc(reset=T)
library(data.table)
library(caret)
library(pbapply)
dat_raw = fread('https://s3.amazonaws.com/datarobot_public_datasets/multiclass/mnist_20.csv')

#Splits
set.seed(42)
dat = copy(dat_raw)
dat <- dat[sample(1:.N),]
train <- dat[1:10000,]
test <- dat[10001:12000,]

train <- head(train, 1000)

train_x <- train[,setdiff(names(train), 'target'), with=F]
test_x <- test[,setdiff(names(train), 'target'), with=F]

zero_var <- pbsapply(train_x, function(x) length(unique(x)))
keep <- which(zero_var > 1)

train_x <- train_x[,keep,with=F]
test_x <- test_x[,keep,with=F]

mod <- prcomp(train_x, center=T, scale=T)
train_x <- data.table(predict(mod, train_x)[,1:12])
test_x <- data.table(predict(mod, test_x)[,1:12])

lvl <- 0:9
lbl <- paste0('d', lvl)
train_y <- train[,factor(target, levels=lvl, labels=lbl)]
test_y <- train[,factor(target, levels=lvl, labels=lbl)]

#Models
#x=data.table(modelLookup()); x[which(forClass & probModel), sort(unique(model))]
model <- train(
  train_x, train_y, method='lda',
  trControl=trainControl(classProbs = T,  verboseIter=T, numer=2)
)

#Predict
pred_good = predict(model, test_x, type='prob')
#pred_good = rowSums(pred_good^2)
pred_good = apply(pred_good, 1, max)
pred_bad = rep(1/length(unique(train_y)), length(pred_good))^2

#Partial dep
bucket <- function(x){
  as.integer(cut(x, quantile(x, 0:10/10), include.lowest = T))
}
pds <- lapply(names(test_x), function(n){
  x = test_x[[n]]
  agg = data.table(
    x=bucket(x),
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
pds = pds[pd == 'good',]
keep = pds[pd == 'good',list(r=diff(range(value))), by='variable'][order(r, decreasing=T),head(variable, 12)]
pds = pds[variable %in% keep,]

ggplot(pds, aes(x=x, y=value, col=pd)) +
  geom_line() + theme_bw() + facet_wrap(~variable) +
  theme(legend.position="top")
