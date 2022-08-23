# Setup
stop()
rm(list=ls(all=T))
gc(reset=T)
library(data.table)
library(readr)
library(jsonlite)
library(compiler)
library(irlba)
library(stringi)
library(pbapply)
library(text2vec)
library(rsparse)
library(datarobot)
library(MetricsWeighted)
ConnectToDataRobot(configPath='~/.config/datarobot/drconfig.yaml')
options("rsparse_omp_threads" = 8)

# Load data dump
dat <- fread('~/R_squared_2018_2021_partial.csv')
setnames(dat, '_id', 'lid')
setnames(dat, 'test.R Squared.0', 'R2_valid')
setnames(dat, 'test.R Squared.1', 'R2_cv')

# Add some newer projects from prod
load_from_dr <- function(pid, metric='R Squared'){
  test_set_raw <- ListModels(pid)
  test_set <- lapply(test_set_raw, function(x){
    data.table(
      pid=pid,
      lid=x[['modelId']],
      dataset_id=pid,
      blueprint_id=x[['blueprintId']],
      R2_valid=x[['metrics']][[metric]][['validation']],
      R2_cv=x[['metrics']][[metric]][['crossValidation']],
      samplesize=x[['trainingRowCount']]
    )
  })
  test_set <- rbindlist(test_set)
  return(test_set)
}
si_lactose <- load_from_dr('602dd1ffbb798c7c18ce56c1')  # si ware lactose
summary(si_lactose)
dat <- rbind(dat, si_lactose, use.names=T, fill=T)

# Subset data
dat <- dat[!is.na(R2_valid),]
dat <- dat[R2_valid < 1,]  # only 38 rows, but need to investigate why
sapply(dat, function(x) sum(is.na(x)))

# Make factors
dat[,pid := factor(pid)]
dat[,dataset_id := factor(dataset_id)]
dat[,blueprint_id := factor(blueprint_id)]

# Make test set
# 60cce1af1fdcadf56d44b027 - si dataset: pure numeric, deep learning - lactose
# 60ee0de225da6ed20c67c9d0 - si ware SNF (should be correlated to lactose)
# 60a6da70d1b6ae84cbdf86ce - common lit, text interpretation
# 60c247e7d52118e3719e0f29 - common lit with roberta embeddings addeds
# 5f2aed9fd6e13f04f6208eba - double pendulum
# 5eb965bf4414d7295713c70c - insurance dataset from json
# 5e3cd501f86f2d11bd249760 - basketball
# 5eb04af1bdede7087c81006b - mercari
test_set <- load_from_dr('5eb04af1bdede7087c81006b', metric='R Squared')
#test_set[!is.na(R2_cv), R2_valid := R2_cv]

# Key the data
keys <- c('pid', 'blueprint_id')
setkeyv(dat, keys)
setkeyv(test_set, keys)

# Aggregate to one result per pid/blueprint
dat_agg <- dat[, list(R2_valid=max(R2_valid)), by=keys]
test_set_agg <- test_set[,list(R2_valid=max(R2_valid)), by=keys]

# Drop dataset/models that are just plain bad
dat_agg <- dat_agg[!is.na(R2_valid),]
test_set_agg <- test_set_agg[!is.na(R2_valid),]

dat_agg <- dat_agg[R2_valid > 0,]
test_set_agg <- test_set_agg[R2_valid > 0,]

#dat_agg[blueprint_id == 'd810e529deea137414c9f96d828a605a',]  # best model in seed project

# L2 normalize
# L2_norm <- function(x){
#   x / sqrt(sum(x**2))
# }
# dat_agg[,R2_valid := L2_norm(R2_valid), by='pid']
# test_set_agg[,R2_valid := L2_norm(R2_valid), by='pid']

# Rank pids by similarity
# neighbors[blueprint_id %in% dat_agg[,sort(unique(blueprint_id))], dput(sort(unique(blueprint_id)))]
test_set_agg[,pid := NULL]
neighbors <- merge(dat_agg, test_set_agg, by='blueprint_id')
neighbors <- neighbors[,list(sim = sum(R2_valid.x * R2_valid.y)), by='pid']
setorderv(neighbors, 'sim')
neighbors <- neighbors[sim>0,]  # We don't want to use dissimilar neighbors for ranking
setorderv(neighbors, 'sim')
neighbors
neighbors[pid=='60cce1af1fdcadf56d44b027',]  # Seed project

# Make a pid/bid to lid map
setorder(dat, -R2_valid)
pidbid_to_lid_map <- dat[,list(lid=lid[1]), by=c('pid', 'blueprint_id')]

# Rank BPs by similarity
# We use mean
# sum is biased towards models we run a lot
key <- 'pid'
setkeyv(neighbors, key)
recs <- merge(dat_agg, neighbors, by=key)
setkeyv(recs, 'blueprint_id')
recs <- recs[,list(score = mean(R2_valid * sim), pid=pid[which.max(sim)]), by='blueprint_id']
recs <- recs[score>0.0001,]
recs <- recs[! blueprint_id %in% test_set[,blueprint_id],]
recs <- merge(recs, pidbid_to_lid_map, by=c('pid', 'blueprint_id'), all.x=T, all.y=F)
recs <- recs[order(score), list(pid, blueprint_id, lid, score)]
tail(recs, 25)
recs[blueprint_id == 'd810e529deea137414c9f96d828a605a',]  # best model in seed project

# TODO: EXCLUDED MODELS FROM RECS
# no blenders
# no scaleout
# no OSS
# etc.

# Map back to blueprint
bp_id <- '"21126d7b2ff7d64bda6713246bf15c8e"'
system(paste('grep -m 1', bp_id, '~/R_squared_2018_2021_partial.json'))
