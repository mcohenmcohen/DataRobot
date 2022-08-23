library(data.table)
library(reshape2)
library(rstanarm)
library(lavaan)
dat <- fread('~/workspace/data-science-scripts/zach/Binary_data_pipelines.csv')
dat <- melt.data.table(dat, id.vars = 'Pipeline', variable.name='dataset')

pipeline_map <- dat[,sort(unique(Pipeline))]
dataset_map <- dat[,sort(unique(dataset))]

dat_test <- dat[is.na(value),]
dat <- dat[!is.na(value),]
setnames(dat, 'value', 'gini')

dat[,dataset := factor(match(dataset, dataset_map))]
dat[,Pipeline := factor(match(Pipeline, pipeline_map))]

mat_dataset <- model.matrix(gini ~ 0 + dataset, dat)
mat_pipeline <- model.matrix(gini ~ 0 + Pipeline, dat)
dat_model <- data.frame(
  gini = dat[,gini],
  mat_dataset,
  mat_pipeline
)

model <- paste(
  paste('f1 =~', paste(colnames(mat_dataset), collapse=' + ')),
  paste('f2 =~', paste(colnames(mat_pipeline), collapse=' + ')),
  paste('gini ~ f1 * f2 + ', paste(colnames(mat_dataset), collapse=' + '), '+', paste(colnames(mat_pipeline), collapse=' + ')),
  'gini ~ 1',
  sep='\n'
)

fit <- sem(model, data=dat_model)
summary(fit, standardized=TRUE)