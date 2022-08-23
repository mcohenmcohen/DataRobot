
########################################################
# Libraries
########################################################

library(data.table)
library(mongolite)
library(jsonlite)
library(pbapply)
library(ggplot2)
library(ggthemes)
rm(list=ls(all=T))
gc(reset=T)

#https://ropensci.org/blog/blog/2017/03/10/mongolite
#https://jeroen.github.io/mongolite/

########################################################
# Useful functions
########################################################

task_info_table <- function(res, important_cols_only=TRUE, transpose=T){
  library(data.table)
  library(mongolite)
  x <- res[['task_info']][[1]][[1]]
  x <- lapply(x, function(out){
    out <- out[!(sapply(out, is.null))]
    out <- data.table(data.frame(out))
    setnames(out, gsub("predict", "transform", names(out)))
    return(out)
  })
  out <- rbindlist(x, fill=TRUE, use.names=TRUE)
  #print(out)
  for(n in c(
    'fit.max.RAM', 'fit.avg.RAM', 'transform.max.RAM', 'fit.total.RAM',
    'transform.avg.RAM', 'transform.total.RAM', 'fit.CPU.time',
    'fit.clock.time', 'transform.clock.time', 'transform.CPU.time',
    'transform.sys.time', 'fit.sys.time')){
    if(! n %in% names(out))
      set(out, j=n, value=as.numeric(NA))
  }

  out[, fit.max.RAM := fit.max.RAM / 1024^3]
  out[, fit.avg.RAM := fit.avg.RAM / 1024^3]
  out[, transform.max.RAM := transform.max.RAM / 1024^3]
  out[, fit.total.RAM := fit.total.RAM / 1024^3]
  out[, transform.avg.RAM := transform.avg.RAM / 1024^3]
  out[, transform.total.RAM := transform.total.RAM / 1024^3]

  out[, fit.CPU.time := fit.CPU.time / 3600]
  out[, fit.clock.time := fit.clock.time / 3600]
  out[, transform.clock.time := transform.clock.time / 3600]
  out[, transform.CPU.time := transform.CPU.time / 3600]
  out[, transform.sys.time := transform.sys.time / 3600]
  out[, fit.sys.time := fit.sys.time / 3600]

  setnames(out, gsub(".RAM", ".RAM.GB", names(out), fixed=TRUE))
  setnames(out, gsub(".time", ".time.hours", names(out), fixed=TRUE))

  #Order cols
  first <- c(
  'task_name', 'fit.max.RAM.GB', 'transform.max.RAM.GB', 'fit.clock.time.hours',
  'transform.clock.time.hours', 'fit.CPU.pct', 'transform.CPU.pct', 'fit.CPU.time.hours',
  'transform.CPU.time.hours', 'fit.sys.time.hours', 'transform.sys.time.hours', 'fit.total.RAM.GB',
  'transform.total.RAM.GB', 'transform.avg.RAM.GB')

  setcolorder(out, c(first, setdiff(names(out), first)))

  # Subset Cols
  if(important_cols_only){
    out = out[,first, with=F]
  }

  # Turn into data.table so we can name rows
  out <- as.data.frame(out)
  row.names(out) <- make.unique(out[['task_name']])
  out[['task_name']] <- NULL

  # Transpose for neater printing
  if(transpose){
    out <- t(out)
  }

  return(out)
}

pull_id <- function(lid, filter=NULL, host, collection){
  library(mongolite)
  con <- mongolite::mongo(
    db = 'MMApp',
    collection = collection,
    url=paste0('mongodb://', host), verbose=F)
  if(is.null(filter)){
    res = con$find(
      query = paste0('{"_id": {"$oid":"', lid, '"}}'),
      limit=1)
  } else {
    res = con$find(
      query = paste0('{"_id": {"$oid":"', lid, '"}}'),
      fields = paste0('{', filter, ', "_id": false}'),
      limit=1)
  }
  rm(con)
  sink <- gc()
  return(res)
}

pull_pid <- function(pid, filter=NULL, host){
  return(pull_id(pid, filter, host, collection='project'))
}

pull_lid <- function(lid, filter=NULL, host){
  return(pull_id(lid, filter, host, collection='leaderboard'))
}

blueprint_info <- function(lid, host, transpose=T){
  res <- pull_lid(lid, '"task_info":1', host)
  return(task_info_table(res, transpose=transpose))
}

blueprint_json <- function(lid, host="10.20.53.43"){
  library(jsonlite)
  res <- pull_lid(lid, '"blueprint":1', host)
  res <- toJSON(res, pretty=TRUE)
  return(res)
}

get_bp_info_json <- function(lid, host="10.20.53.43"){
  print(blueprint_json(lid, host))
  print(blueprint_info(lid, host))
}

lid_info <- function(lid, host){
  f <- '"pid":1, "lid":1, "samplepct":1, "samplesize":1, "total_size":1, "model_type":1'
  res <- pull_lid(lid, f,  host)
  name <- pull_pid(res$pid, '"originalName":1',  host)
  return(cbind(name, res))
}

lid_and_bp_info <- function(lid, host){
  lid_out <- lid_info(lid, host)
  bp_out <- blueprint_info(lid, host, transpose=F)
  bp_out <- cbind(data.table(task=row.names(bp_out)), bp_out)
  out <- cbind(lid_out, bp_out)
  return(out)
}

OWENS_WORLD_IP <- '10.20.55.207'


########################################################
# Slow keras model on prod
########################################################

bp <- '5ec7e37cf3ab983e4cad611a'
blueprint_json(bp, host="mongo-bi.int.datarobot.com")
blueprint_info(bp, host="shrink-mongo-0.infra.ent.datarobot.com:27017")

########################################################
# 6.1 mbtest
########################################################

# fastiron 6.0 - tfnn - 16%
blueprint_json('5e6d9e5c27c12327b0e1b3ff', host="shrink-mongo-0.infra.ent.datarobot.com:27017")
blueprint_info('5e6d9e5c27c12327b0e1b3ff', host="shrink-mongo-0.infra.ent.datarobot.com:27017")

# fastiron 6.1 - tfnn - 16%
blueprint_json('5e964c0b4364b4217452908b', host="shrink-mongo-0.infra.ent.datarobot.com:27017")
blueprint_info('5e964c0b4364b4217452908b', host="shrink-mongo-0.infra.ent.datarobot.com:27017")

########################################################
# 6.1 mbtest
########################################################

# Airbnb 6.0
blueprint_json('5e6d85138593c61215f94c1a', host="shrink-mongo-0.infra.ent.datarobot.com:27017")
old = blueprint_info('5e6d85138593c61215f94c1a', host="shrink-mongo-0.infra.ent.datarobot.com:27017")

# Airbnb 6.1
blueprint_json('5e964a6db2cb8e197a813f2b', host="shrink-mongo-0.infra.ent.datarobot.com:27017")
new = blueprint_info('5e964a6db2cb8e197a813f2b', host="shrink-mongo-0.infra.ent.datarobot.com:27017")

round(old - new, 2)[3,,drop=F]
round(old - new, 2)[4,,drop=F]

round(old - new, 2)[,'ESXGBR2',drop=F]

round(old, 2)[,'ESXGBR2',drop=F]
round(new, 2)[,'ESXGBR2',drop=F]

########################################################
# Owen's world 10GB OTV Projects - round 2
########################################################
# SGD with fix
LID <- '5e5619dd34c997ba60eb19a0'
blueprint_json(LID, host=OWENS_WORLD_IP)
round(blueprint_info(LID, host=OWENS_WORLD_IP), 2)

# GBM
LID <- '5e50368a34c9978c2424ec7a'
blueprint_json(LID, host=OWENS_WORLD_IP)
round(blueprint_info(LID, host=OWENS_WORLD_IP), 2)

#  XGBoost
LID <- '5e5039cb34c9978c2424ec85'
blueprint_json(LID, host=OWENS_WORLD_IP)
round(blueprint_info(LID, host=OWENS_WORLD_IP), 2)

# TF
LID <- '5e50365d34c9978c2424ec69'
blueprint_json(LID, host=OWENS_WORLD_IP)
round(blueprint_info(LID, host=OWENS_WORLD_IP), 2)

########################################################
# Owen's world 10GB OTV Projects
########################################################

# Generalized Additive2 Model
LID <- '5e4dcea234c997b58ecaf3e1'
blueprint_json(LID, host=OWENS_WORLD_IP)
round(blueprint_info(LID, host=OWENS_WORLD_IP), 2)

#  Auto-tuned Stochastic Gradient Descent Regression
LID <- '5e4dcea234c997b58ecaf3e6'
blueprint_json(LID, host=OWENS_WORLD_IP)
round(blueprint_info(LID, host=OWENS_WORLD_IP), 2)

########################################################
# Regression in XGboost on YARN
########################################################

new = blueprint_info('5d7cde8ae2d574340385f596', host="shrink-mongo-0.infra.ent.datarobot.com:27017")
old = blueprint_info('5ddd3e8cb879a318ea06f313', host="shrink-mongo-0.infra.ent.datarobot.com:27017")

round(new, 4)
round(old, 4)
round(new / old, 4)

########################################################
# Regression in XGboost on YARN
########################################################

# 2019-11-27
new = blueprint_info('5ddf7923b5ddc16cb83b08bb', host="shrink-mongo-0.infra.ent.datarobot.com:27017")

# 2019-09-04
old = blueprint_info('5d70aed334b97f38b0fdfb40', host="shrink-mongo-0.infra.ent.datarobot.com:27017")

round(new, 4)
round(old, 4)
round(new / old, 4)

########################################################
# Prod Xgboost
########################################################

# Kdd, 2019-12-11
xgb = blueprint_info('5df17e953ba29701fba6340e', host="mongo-bi.int.datarobot.com")
round(xgb, 4)

lgbm = blueprint_info('5df17e963ba29701fba63410', host="mongo-bi.int.datarobot.com")
round(lgbm, 4)


# 10k lending club 2019-11-20
new = blueprint_info('5dd5a119f403c6025886624e', host="mongo-bi.int.datarobot.com")
round(new, 4)

# 10k lending club 2019-09-03
old = blueprint_info('5d6ed6ce27f88e4a2f044e3a', host="mongo-bi.int.datarobot.com")
round(old, 4)
round(new / old, 4)




# 10k lending club 2019-12-12
new = blueprint_info('5df2b7293ba2975f58a6709c', host="mongo-bi.int.datarobot.com")
round(new, 4)

# 10k lending club 2019-09-03
old = blueprint_info('5d6ed6ce27f88e4a2f044e45', host="mongo-bi.int.datarobot.com")
round(old, 4)
round(new / old, 4)



#  PXGBC2  ???
# farmers roof 2019-10-03
old = blueprint_info('5d95f92a30892dfda2d2eb35', host="mongo-bi.int.datarobot.com")
round(old, 4) #- 1 cores

# aircraft images roof 2019-10-01
old = blueprint_info('5d9354fac491ee438e773b41', host="mongo-bi.int.datarobot.com")
round(old, 4) #- 1.15 cores?



# mer text combo - 2019-10-04
old = blueprint_info('5d9e1cdf69de30578e08b802', host="mongo-bi.int.datarobot.com")
round(old, 4) # 1 cores


# inside_past3_daily_opps_with_targets.csv on 2019-10-04 - Alex? 23:37
old = blueprint_info('5d9815991c212f03b0a0f704', host="mongo-bi.int.datarobot.com")
round(old, 4) # 1 cores

# AML_alerts.csv on 2019-10-03 - Jon?
old = blueprint_info('5d966c853f52201ea3032ce4', host="mongo-bi.int.datarobot.com")
round(old, 4) # 1 cores




# EXGB
# 10K_Lending_Club_Loans.csv 2019-10-04
old = blueprint_info('5d974d573f52203341032d00', host="mongo-bi.int.datarobot.com")
round(old, 4) #1 cores?

# vanga-v1-1 2019-10-03
old = blueprint_info('5d963944106b58778baf375a', host="mongo-bi.int.datarobot.com")
round(old, 4) #1 cores?

# novaris train 2019-09-26
old = blueprint_info('5d8d3105fd38f3005224a28d', host="mongo-bi.int.datarobot.com")
round(old, 4) #- 4 cores?

# Avito 5GB on 2019-09-25
old = blueprint_info('5d8ca00b843d0a0e61f55b91', host="mongo-bi.int.datarobot.com")
round(old, 4) #- 4 cores




# Shrink test of fix
# 4 CPUs
round(blueprint_info('5df34c68112cd2d8c4e66181', host="shrink-mongo-0.infra.ent.datarobot.com:27017"), 4)


########################################################
# OTV auto vs fast hist XGB
########################################################

blueprint_json('5d11d90f6b651d0dc1a01690', host="mongo-bi.int.datarobot.com")
blueprint_json('5d95c77f45eebd1fc26723f8', host="mongo-bi.int.datarobot.com")

########################################################
# OTV auto vs fast hist XGB
########################################################

x = '5d9bc9b262d3cf1e00e226ef'
blueprint_json(x, host="mongo-bi.int.datarobot.com")
pull_lid(x, '"task_info":1', "mongo-bi.int.datarobot.com")

########################################################
# OTV auto vs fast hist XGB
########################################################

round(blueprint_info('5d3277914d8976582d486423', host="mongo-bi.int.datarobot.com"), 2)
round(blueprint_info('5d3f62b477a90f162484a3df', host="mongo-bi.int.datarobot.com"), 2)

########################################################
# OOMs on prod (pull data from smaller sample)
########################################################

# default
blueprint_json('5d2d3b900d027b7d8c469b68', host="mongo-bi.int.datarobot.com")
round(blueprint_info('5d2d3b900d027b7d8c469b68', host="mongo-bi.int.datarobot.com"), 2)

# fast hist
blueprint_json('5d3f3eb077a90f0ffe84a4c5', host="mongo-bi.int.datarobot.com")
round(blueprint_info('5d3f3eb077a90f0ffe84a4c5', host="mongo-bi.int.datarobot.com"), 2)

########################################################
# OOMs on prod (pull data from smaller sample)
########################################################

# default
blueprint_json('5d3f1d9177a90f0a3c84a5e5', host="mongo-bi.int.datarobot.com")
round(blueprint_info('5d3f1d9177a90f0a3c84a5e5', host="mongo-bi.int.datarobot.com"), 2)

# fast hist
blueprint_json('5d3f3ec877a90f0fbe84a683', host="mongo-bi.int.datarobot.com")
round(blueprint_info('5d3f3ec877a90f0fbe84a683', host="mongo-bi.int.datarobot.com"), 2)

########################################################
# Keras models
########################################################

blueprint_json('5d273ed20d027b06eb469919', host="mongo-bi.int.datarobot.com")
round(blueprint_info('5d273ed20d027b06eb469919', host="mongo-bi.int.datarobot.com"), 2)

########################################################
# 100GB manual test
########################################################

# 14% LGBM that worked
blueprint_json('5ce69552c50f7b247eed4f45', host="hpc-lab2.hq.datarobot.com:27017")
round(blueprint_info('5ce69552c50f7b247eed4f45', host="hpc-lab2.hq.datarobot.com:27017"), 2)

# 7% keras that worked
blueprint_json('5ce6991cc50f7b247eed4fb4', host="hpc-lab2.hq.datarobot.com:27017")
round(blueprint_info('5ce6991cc50f7b247eed4fb4', host="hpc-lab2.hq.datarobot.com:27017"), 2)

# 4% enet that wroked
blueprint_json('5ce62847c50f7b4033acfd20', host="hpc-lab2.hq.datarobot.com:27017")
round(blueprint_info('5ce62847c50f7b4033acfd20', host="hpc-lab2.hq.datarobot.com:27017"), 2)

########################################################
# 100GB manual test
########################################################

blueprint_json('5ce6991cc50f7b247eed4fb3', host="hpc-lab2.hq.datarobot.com:27017")
round(blueprint_info('5ce6991cc50f7b247eed4fb3', host="hpc-lab2.hq.datarobot.com:27017"), 2)

########################################################
# 100GB manual test - keras - no smart sampled
########################################################

lids_10 <- c(
  '5caf643dc50f7b88de9ab504',
  '5caf643ec50f7b88de9ab506',
  '5caf643ec50f7b88de9ab508',
  '5caf643ec50f7b88de9ab505',
  '5caf643ec50f7b88de9ab507'
)

lids_98 <- c(
  '5caf859ac50f7b6d1cfeab3d',
  '5caf8318c50f7b6d1cfeab19',
  '5caf8312c50f7b6d1cfeab0b',
  '5caf831cc50f7b6d1cfeab20'
)

# Pull all data
out_10 <- lapply(
  lids_10, blueprint_info, host='hpc-lab2.hq.datarobot.com:27017')
out_98 <- lapply(
  lids_98, blueprint_info, host='hpc-lab2.hq.datarobot.com:27017')

# Lookit results
lapply(out_10, round, 2)
lapply(out_98, round, 2)

########################################################
# 100GB manual test
########################################################

lids_50 <- c(
  '5ca3dbc725752f63e75ea0d2',
  '5ca3dbcb25752f63e75ea0da',
  '5ca3dbcf25752f63e75ea0e2'
)

lids_75 <- c(
  '5ca3dbd825752f63e75ea0f2',
  '5ca3dbd425752f63e75ea0ea',
  '5ca3dbdc25752f63e75ea0fa'
)

lids_98 <- c(
  '5ca3dbea25752f63e75ea10a',
  '5ca3dbe625752f63e75ea102',
  '5ca3dbee25752f63e75ea112'
)

# Pull all data
out_50 <- lapply(
  lids_50, blueprint_info, host='hpc-lab1.hq.datarobot.com:27017')
out_75 <- lapply(
  lids_75, blueprint_info, host='hpc-lab1.hq.datarobot.com:27017')
out_98 <- lapply(
  lids_98, blueprint_info, host='hpc-lab1.hq.datarobot.com:27017')

# Lookit results
lapply(out_50, round, 2)
lapply(out_75, round, 2)
lapply(out_98, round, 2)

########################################################
# 100GB manual test - keras - smart sampled
########################################################

lids_33 <- c(
  '5ca87326c50f7bc510d0f2b8',
  '5ca87326c50f7bc510d0f2ba',
  '5ca87327c50f7bc510d0f2bb',
  '5ca87326c50f7bc510d0f2b9'
)

lids_98 <- c(
  '5cab4c8fc50f7b56d20711ed',
  '5cab4c98c50f7b56d20711fb',
  '5cab4c93c50f7b56d20711f4'
  #'5cab4c89c50f7b56d20711e6'
)

# Pull all data
out_33 <- lapply(
  lids_33, blueprint_info, host='hpc-lab2.hq.datarobot.com:27017')
out_98 <- lapply(
  lids_98, blueprint_info, host='hpc-lab2.hq.datarobot.com:27017')

# Lookit results
lapply(out_33, round, 2)
lapply(out_98, round, 2)

########################################################
# 100GB manual test
########################################################

all_lids <- c(
  '5c9fafb325752f8a265c04bf',
  '5c9fef8725752f63e75e9f64',
  '5c9fef8c25752f63e75e9f6c',
  '5c9fef8125752f63e75e9f5c'
)

# Pull all data
out_list <- lapply(
  all_lids, blueprint_info, host='hpc-lab1.hq.datarobot.com:27017')

# Lookit results
lapply(out_list, round, 2)

########################################################
# Long-running keras models
########################################################

blueprint_json('5c89ee8b2514724203c27e08', host="shrink-mongo-0.infra.ent.datarobot.com:27017")
round(blueprint_info('5c89ee8b2514724203c27e08', host="shrink-mongo-0.infra.ent.datarobot.com:27017"), 2)

########################################################
# 100GB Mbtest
########################################################

# https://datarobot.atlassian.net/browse/MODEL-438
# https://datarobot.atlassian.net/browse/MODEL-434

all_lids <- c(
  '5c73dcdff4f60701fc430d4b',
  '5c73dd1af4f60701fc430e0f',
  '5c73dce4f4f60701fc430d7d',
  '5c73dce9f4f607020b430da0',
  '5c75431bf4f6070369430e4d',
  '5c7541d7f4f6070369430d5b',
  '5c7541e7f4f607034b430d25',
  '5c7541d1f4f6070369430d50',
  '5c7541d1f4f6070369430d4b',
  '5c754236f4f6070369430da8',
  '5c753cf7e38238372b05d1da',
  '5c754220f4f6070369430d8b',
  '5c75422df4f6070369430d99',
  '5c75428df4f607034b430d64',
  '5c75427cf4f607034b430d5d',
  '5c75431bf4f6070369430e4d',
  '5c7542bcf4f607034b430d79',
  '5c7542c8f4f607034b430d87',
  '5c7542dcf4f6070369430e27'
)

# Pull all data
out_list <- pblapply(
  all_lids, lid_and_bp_info, host='shrink-mongo-0.infra.ent.datarobot.com:27017')

# Combine and save it
out <- rbindlist(out_list)
out[,task := gsub('\\.[[:digit:]]+$', '', task)]
fwrite(out, '~/datasets/100GB_results.csv')

# Plots
ggplot(out, aes(x=task, y=fit.max.RAM.GB)) + 
  geom_boxplot() + theme_tufte() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

ggplot(out, aes(x=task, y=transform.max.RAM.GB)) + 
  geom_boxplot() + theme_tufte() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

ggplot(out, aes(x=task, y=fit.clock.time.hours)) + 
  geom_boxplot() + theme_tufte() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

ggplot(out, aes(x=task, y=transform.clock.time.hours)) + 
  geom_boxplot() + theme_tufte() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

########################################################
# 100GB
########################################################

# 768 GB server - outbrain - regression
# 5c6d71ad4b8830278f6cd3a7 - 30% Random Forest
# 5c6d71a24b8830278f6cd397 - 30% LGBM
# 5c6d71b24b8830278f6cd3af - 5% Ridge + constant splines (GS)
# 5c6949804b88301a1ebf3c0d - 10% Ridge + smooth ridit
# 5c6949814b88301a1ebf3c0f - 10% Ridge + standardize
round(blueprint_info('5c6d71ad4b8830278f6cd3a7', host="10.50.225.209"), 2)
round(blueprint_info('5c6d71a24b8830278f6cd397', host="10.50.225.209"), 2)
round(blueprint_info('5c689ffa4b8830ea21bf3c12', host="10.50.225.209"), 2)
round(blueprint_info('5c6949804b88301a1ebf3c0d', host="10.50.225.209"), 2)
round(blueprint_info('5c6949814b88301a1ebf3c0f', host="10.50.225.209"), 2)

# LGBM on dev server (488 GB) - outbrain
# 60% 5c601d8445058870ca0106d1
# 80% 5c60927f45058870ca010760 
round(blueprint_info('5c60927f45058870ca010760', host="10.50.225.196"), 2)

# LGBM on thrust (256GB) - outbrain
# 15%? 5c586fdc9636a6280eb0a7e1
# 32% - 5c59f1f19636a6280eb0a8c6
round(blueprint_info('5c59f1f19636a6280eb0a8c6', host="10.20.40.30"), 2)

# XGB on dev instance (48*GB) - Glen 100GB simulated numeric no missing
# 10% - 5c584ebd4b2efa70c9617dc5
# 30% - 5c58557c4b2efa70c9617ddb
#blueprint_json('5c584ebd4b2efa70c9617dc5', host="10.50.225.141")
round(blueprint_info('5c584ebd4b2efa70c9617dc5', host="10.50.225.141"), 2)
round(blueprint_info('5c58557c4b2efa70c9617ddb', host="10.50.225.141"), 2)

########################################################
# Prod slow model
########################################################

# epoch size = Full n_iter=10,000
blueprint_json('5c88055f79bffe6c982e17af', host="mongo-bi.int.datarobot.com")
round(blueprint_info('5c88055f79bffe6c982e17af', host="mongo-bi.int.datarobot.com"), 2)

########################################################
# LGBM increase
########################################################

# Old LGBM
blueprint_json('5b919fed46343837a857be6e', host="shrink-mongo-0.infra.ent.datarobot.com:27017")
round(blueprint_info('5b919fed46343837a857be6e', host="shrink-mongo-0.infra.ent.datarobot.com:27017"), 2)

# New LGBM
blueprint_json('5bf364009af7f038c2b24da1', host="shrink-mongo-0.infra.ent.datarobot.com:27017")
round(blueprint_info('5bf364009af7f038c2b24da1', host="shrink-mongo-0.infra.ent.datarobot.com:27017"), 2)

########################################################
# Prod ENET
########################################################

#blueprint_json('5b65e8c0e4e5576cb165f13a', host="mongo-bi.int.datarobot.com")
round(blueprint_info('5b65e8c0e4e5576cb165f13a', host="mongo-bi.int.datarobot.com"), 2)

########################################################
# Prod TF
########################################################

# epoch size = Full n_iter=10,000
#blueprint_json('5b461ae15feaa70888d12c41', host="mongo-bi.int.datarobot.com")
round(blueprint_info('5b461ae15feaa70888d12c41', host="mongo-bi.int.datarobot.com"), 2)

# epoch size = 10,000 n_iter=100
#blueprint_json('5b461a055feaa70828cc4b59', host="mongo-bi.int.datarobot.com")
round(blueprint_info('5b461a055feaa70828cc4b59', host="mongo-bi.int.datarobot.com"), 2)

########################################################
# Jett 100 class multiclass on mbtest
########################################################

# Ridge Regressor with Forecast Distance Modeling
blueprint_json('5b45157e45c84e4872e94dcf', host="shrink-mongo-0.infra.ent.datarobot.com:27017")
round(blueprint_info('5b45157e45c84e4872e94dcf', host="shrink-mongo-0.infra.ent.datarobot.com:27017"), 2)

########################################################
# Jett 100 class multiclass on OW
########################################################

# Ridge Regressor with Forecast Distance Modeling
blueprint_json('5e4567667c2feb016f4c7141', host="10.50.183.145")
round(blueprint_info('5b294e9234c9972cbf02cf8e', host="10.20.53.43"), 2)

########################################################
# time series 7 FD on Owen's Workd
########################################################
# Ridge Regressor with Forecast Distance Modeling
blueprint_json('5b3a352034c99728b51de444', host="10.20.53.43")
round(blueprint_info('5b3a352034c99728b51de444', host="10.20.53.43"), 2)

#  RandomForest Regressor
blueprint_json('5b3a351f34c99728b51de401', host="10.20.53.43")
round(blueprint_info('5b3a351f34c99728b51de401', host="10.20.53.43"), 2)

# Baseline Predictions Using Most Recent Value
blueprint_json('5b3a352034c99728b51de463', host="10.20.53.43")
round(blueprint_info('5b3a352034c99728b51de463', host="10.20.53.43"), 2)

# eXtreme Gradient Boosting on ElasticNet Predictions
blueprint_json('5b3a351f34c99728b51de400', host="10.20.53.43")
round(blueprint_info('5b3a351f34c99728b51de400', host="10.20.53.43"), 2)

########################################################
# time series 1 FD on staging
########################################################
blueprint_json('5b33d392a965ebc4b43b4d51', host="staging-mongo-0.infra.ent.datarobot.com")
round(blueprint_info('5b33d392a965ebc4b43b4d51', host="staging-mongo-0.infra.ent.datarobot.com"), 2)

########################################################
# VW on OW
########################################################
blueprint_json('5ad710e134c9978238f5c48d', host="10.20.53.43")
round(blueprint_info('5ad710e134c9978238f5c48d', host="10.20.53.43"), 2)

blueprint_json('5ad6a6e634c99713c1f5c48a', host="10.20.53.43")
########################################################
# Release 4.2 Mbtest issues
########################################################

# 4.0
get_bp_info_json('59f2de1356951303b7cf53f2', host="shrink-mongo.dev.hq.datarobot.com")

# 4.2
get_bp_info_json('5a7ee6a34b497805cd116a74', host="shrink-mongo-0.infra.ent.datarobot.com:27017")

########################################################
# Ownen's World Ridge Regressor OOM
########################################################

#16
get_bp_info_json('5a67b5e834c9977dc2624977')

#32
get_bp_info_json('5a67b5de34c9977dc2624970')

#64
get_bp_info_json('5a67b59d34c9977dc2624958')

########################################################
# Owen's World GA2M RAM Usage
########################################################

get_bp_info_json('5a00af7634c997963fe8a8fd')

########################################################
# Owen's World test YS-L's XGB V2 Rulefit RAM usage
########################################################

#Rulefit
get_bp_info_json('5992151b34c9970c5ed0ff8e')

#GBM
get_bp_info_json('5992151e34c9970c5ed0ff91')

#XGB + unsup
get_bp_info_json('5992151e34c9970c5ed0ff90')

########################################################
# Owen's World Mandy's Anomaly detection test
########################################################

#LOF - DR_Demo
get_bp_info_json('598333d034c99711b2c9c493')

#Double MAD - GSOD
get_bp_info_json('5983477434c997715bc9c493')
get_bp_info_json('5992151e34c9970c5ed0ff90')

########################################################
# MBtest Eureqa bluepriunt
########################################################

get_bp_info_json('59cf203b713a2e10cb7881b7', host="shrink-mongo.dev.hq.datarobot.com")

# Back to the future
round(blueprint_info('5e4880f720118a714c1b6c2b', host="shrink-mongo-0.infra.ent.datarobot.com:27017"), 2)
blueprint_json('5e4567667c2feb016f4c7141', host="shrink-mongo-0.infra.ent.datarobot.com:27017")

round(blueprint_info('5df2b6883ba2975d89a6706a', host="mongo-bi.int.datarobot.com"), 2)
