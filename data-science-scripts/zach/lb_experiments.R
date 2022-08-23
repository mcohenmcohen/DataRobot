# Setup
rm(list=ls(all=T))
gc(reset=T)
library(data.table)
library(readr)
library(jsonlite)
library(compiler)
library(datarobot)
library(irlba)
library(stringi)
library(pbapply)
library(text2vec)
library(rsparse)
ConnectToDataRobot(configPath='~/.config/datarobot/drconfig.yaml')
options("rsparse_omp_threads" = 8)

# Load and subset data
dat <- fread('~/R_squared_2018_2021_partial.csv')
setnames(dat, '_id', 'lid')
setnames(dat, 'test.R Squared.0', 'R2_valid')
setnames(dat, 'test.R Squared.1', 'R2_cv')
dat <- dat[!is.na(R2_valid),]
dat <- dat[R2_valid < 1,]  # only 38 rows, but need to investigate why
dat <- dat[R2_valid > 0,]  # drop bad models
sapply(dat, function(x) sum(is.na(x)))

# Make factors
dat[,pid := factor(pid)]
dat[,dataset_id := factor(dataset_id)]
dat[,blueprint_id := factor(blueprint_id)]

# Normalize R2
setkeyv(dat, c('pid', 'blueprint_id'))
dat[,R2_valid_min := min(R2_valid), by='pid']
dat[R2_valid < 0, R2_valid := -1 * R2_valid / R2_valid_min]
dat[,R2_valid_min := NULL]

# Add some newer projects to learn from
load_from_dr <- function(pid, return_data_table=F){
  test_set_raw <- ListModels(pid)
  test_set <- lapply(test_set_raw, function(x){
    data.table(
      pid=pid,
      dataset_id=pid,
      blueprint_id=x[['blueprintId']],
      R2_valid=x[['metrics']][['R Squared']][['validation']],
      R2_cv=NA,
      samplesize=x[['trainingRowCount']]
    )
  })
  test_set <- rbindlist(test_set)
  summary(test_set)
  test_set[,R2_valid_min := min(R2_valid), by='pid']
  test_set[R2_valid < 0, R2_valid := -1 * R2_valid / R2_valid_min]
  test_set[,R2_valid_min := NULL]
  setkeyv(test_set, 'blueprint_id')
  if(return_data_table){
    return(test_set)
  }
  test_set <- merge(test_set, blueprint_map, by=c('blueprint_id'), all.x=F, all.y=F)
  mat_test <- test_set[,sparseMatrix(i=rep(1L, .N),  j=as.integer(id), x=R2_valid, dims=c(1, ncol(mat)))]
  stopifnot(ncol(mat_test) == ncol(mat))
  return(mat_test)
}
si_lactose <- load_from_dr('60cce1af1fdcadf56d44b027', return_data_table=T)  # si ware lactose
summary(si_lactose)

# Join new data to old data
dat <- rbind(dat, si_lactose, use.names=T, fill=T)

# Bucket sample size
dat[,sample_bucket := as.integer(factor(samplesize, levels=rev(sort(unique(samplesize))))), by='pid']

# Make a ranking problem
dat[,rank := rank(1-R2_valid), by='pid']

# Make a matrix
# dat_agg <- dat[,R2_valid := mean(R2_valid), by=c('pid', 'blueprint_id')]  # don't need for pid/bid
dat_agg <- dat[,list(R2_valid=mean(R2_valid)), by=c('pid', 'blueprint_id')]
mat <- dat_agg[,sparseMatrix(i=as.integer(pid),  j=as.integer(blueprint_id), x=R2_valid)]
summary(mat@x)

# Make a blueprint map
blueprint_map <- dat[,list(blueprint_id = as.character(blueprint_id), id = as.integer(blueprint_id))]
setkeyv(blueprint_map, names(blueprint_map))
blueprint_map <- unique(blueprint_map)
stopifnot(nrow(blueprint_map) == ncol(mat))

# Run SVD
# model <- irlba(mat, nv=5, nu=50, work=500, verbose=T, center=centers)
# d <- Diagonal(x=model$d)

# Run a fancier model
model = LinearFlow$new(
  rank = 100, lambda = 0,
  solve_right_singular_vectors = "svd"
)
user_emb = model$fit_transform(mat)

# Make test set
# 60cce1af1fdcadf56d44b027 - si dataset: pure numeric, deep learning - lactose
# 60ee0de225da6ed20c67c9d0 - si ware SNF (should be correlated to lactose)
# 60a6da70d1b6ae84cbdf86ce - common lit, text interpretation
# 5f2aed9fd6e13f04f6208eba - double pendulum
# 5eb965bf4414d7295713c70c - insurance dataset from json
mat_test <- load_from_dr('60ee0de225da6ed20c67c9d0')  # si ware lactose
summary(mat_test@x)

# Predict test set
pred_test <- copy(blueprint_map)
K=1
preds = model$predict(mat_test, k = K, not_recommend = mat_test)[1,]
pred_test <- pred_test[id %in% preds[K],]

# Map back to blueprint
bp_id <- pred_test[.N, paste0('"', blueprint_id, '"')]
# bp_id <- '"e1e6a11d7b26d14ab37d4c53b48e7874"'
system(paste('grep -m 1', bp_id, '~/R_squared_2018_2021_partial.json'))