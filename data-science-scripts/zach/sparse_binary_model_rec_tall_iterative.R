stop()
rm(list=ls(all=T))
gc(reset = T)
library(data.table)
library(reshape2)
set.seed(42)
dat <- fread('~/workspace/data-science-scripts/zach/Binary_data_pipelines.csv')
dat <- melt.data.table(dat, id.vars = 'Pipeline', variable.name='dataset', value.name = 'gini')

rmse <- function(pred, act) sqrt(mean((pred-act)^2, na.rm=T))

pipeline_map <- dat[,sort(unique(Pipeline))]
dataset_map <- dat[,sort(unique(dataset))]

dat[,dataset := factor(match(dataset, dataset_map))]
dat[,Pipeline := factor(match(Pipeline, pipeline_map))]

dat[,set := 'UNKNOWN']
dat[is.na(gini),set := 'pred']
dat[!is.na(gini),set := 'train']
dat[sample(which(set == 'train'), .20*sum(set == 'train')), set := 'test']

# Initialization
dat[,PC1 := rep(0, .N)]
dat[,PC2 := rep(0, .N)]

# Warmup model
model <- lm(gini ~ dataset, data=dat[set=='train',])
dat[,pred := predict(model, dat)]
print(paste('warmup', dat[set=='test',round(rmse(pred, gini), 4)]))

for(i in 1:5){
  print(paste('iteration', i))
  
  # Step 1: PCA residual model
  dat[,gini_resid := gini - pred]
  to_mat <- function(x){
    mat <- dcast.data.table(dat[set=='train',], Pipeline ~ dataset, value.var=x, drop=FALSE)
    mat[,Pipeline := NULL]
    return(as.matrix(mat))
  }
  mat <- to_mat('gini_resid')
  mat_imp <- to_mat('pred')
  idx <- is.na(mat)
  #mat[idx] <- mat_imp[idx]
  mat[idx] <- runif(sum(idx), -1, 1)

  pca_model <- svd(mat)
  
  dat[,PC1 := PC1 + pca_model$u[as.integer(Pipeline), 1] * pca_model$v[as.integer(dataset), 1]]
  dat[,pred := pred + PC1]
  print(paste('...', dat[set=='test',round(rmse(pred, gini), 4)]))
  
  dat[,PC2 := PC2 + pca_model$u[as.integer(Pipeline), 2] * pca_model$v[as.integer(dataset), 2]]
  dat[,pred := pred + PC2]
  print(paste('...', dat[set=='test', round(rmse(pred, gini), 4)]))
  
  # Step 2: main model
  model <- lm(gini ~ dataset + PC1, data=dat[set=='train',])
  dat[,pred := predict(model, dat)]
  print(paste('...', dat[set=='test',round(rmse(pred, gini), 4)]))
}

# Lookit results
dat[,best := 1:.N == which.max(pred), by=c('dataset', 'set')]
dat[best == T & set == 'pred',]
x <- dat[best == T,sort(table(Pipeline))]; print(x[x>0])

pipeline_map[c(877, 1377)]
pipeline_map[c(1413, 877, 1402, 1298, 1349)]
pipeline_map[c(877, 1402, 637)]
