rm(list=ls(all=TRUE))
gc(reset=TRUE)
library(caret)
library(yaml)
library(pbapply)
library(data.table)
library(stringi)

#Decide how big to generate data
ns <- 10^(2:6)
ns <- c(ns, ns*5)
ns <- sort(ns)
outdir <- '~/Desktop/s3files/'
overwrite <- FALSE

#Class
class_models <- lapply(ns, function(n){
  set.seed(43205)
  name <- paste0('caret_sim_binary_n', format(n, scientific=FALSE), '.csv')
  out <- paste0(outdir, name)
  if(overwrite | (!file.exists(out))){
    dat <- twoClassSim(
      n = n, intercept = -5,
      linearVars = 10, noiseVars = 3, corrVars = 3,
      corrType = "AR1", corrValue = .95)
    stopifnot(nrow(dat) == n)
    stopifnot('Class' %in% names(dat))
    stopifnot(!anyNA(dat))
    write.csv(dat, row.names=FALSE, file=out)
    print(dim(dat))
  }

  out <- list(
    dataset_name=paste0("http://test:WtiDBBJWkKjZZwU7@52.4.14.137/datasets/test_data/", name),
    metric='LogLoss',
    target='Class'
  )

  if(n >= 500000){
    out$worker_size <- '>30'
  }

  return(out)
})

#Reg
reg_models <- lapply(ns, function(n){
  set.seed(96005)
  name <- paste0('caret_sim_reg_n', format(n, scientific=FALSE), '.csv')
  out <- paste0(outdir, name)
  if(overwrite | (!file.exists(out))){
    dat <- SLC14_1(
      n = n, noiseVars = 3, corrVars = 3,
      corrType = "AR1", corrValue = .95)
    stopifnot(nrow(dat) == n)
    stopifnot('y' %in% names(dat))
    stopifnot(!anyNA(dat))
    write.csv(dat, row.names=FALSE, file=out)
    print(dim(dat))
  }

  out <- list(
    dataset_name=paste0("http://test:WtiDBBJWkKjZZwU7@52.4.14.137/datasets/test_data/", name),
    metric='RMSE',
    target='y'
  )

  if(n >= 500000){
    out$worker_size <- '>30'
  }

  return(out)
})

all_models <- c(class_models, reg_models)
model_yaml <- as.yaml(all_models)
#cat(model_yaml)
if(overwrite){
  cat(model_yaml, file=paste0(
    '~/workspace/DataRobot/tests/ModelingMachine/',
    'sim_data_num_no_miss_up_to_5M.yaml'
  ))
}

#Make dirtier data
n_low_card <- 5
n_high_card <- 2
n_text <- 2
pct_missing <- .05
overwrite <- FALSE

dirty_datasets <- pblapply(all_models, function(d){
  set.seed(53)
  d$dataset_name <- stri_replace_all_fixed(
    d$dataset_name,
    'http://test:WtiDBBJWkKjZZwU7@52.4.14.137/datasets/test_data/',
    ''
  )
  in_name <- paste0(outdir, d$dataset_name)
  out_name <- paste0('dirty_', d$dataset_name)
  out <- paste0(outdir, out_name)
  dat <- fread(in_name)

  if(overwrite | (!file.exists(out))){
    print('Loading Data')

    print('Adding low card irrelevant')
    for(i in 1:n_low_card){
      n <- 1:sample(2:5, 1)
      levels <- letters[n]
      x <- sample(levels, nrow(dat), replace=TRUE)
      set(dat, j=paste0('Lowcard', i), value=x)
    }

    print('Adding high card irrelevant')
    all_possible <- CJ(letters, letters, letters)
    all_possible <- paste0(all_possible$V1, all_possible$V2, all_possible$V3)
    for(i in 1:n_high_card){
      n <- 1:sample(100:1000, 1)
      levels <- all_possible[n]
      x <- sample(levels, nrow(dat), replace=TRUE)
      set(dat, j=paste0('Highcard', i), value=x)
    }

    print('Adding text irrelevant')
    words <- sapply(1:1000, function(x){
      paste(sample(letters, sample(1:7, 1)), collapse='')
    })
    for(i in 1:n_high_card){
      phrases <- pbsapply(1:nrow(dat), function(x){
        paste(sample(words, sample(1:3, 1)), collapse=' ')
      })
      set(dat, j=paste0('Text', i), value=phrases)
    }

    print('Adding missing data to X and Y')
    for(col in colnames(dat)){
      na <- sample(1:nrow(dat), nrow(dat) * pct_missing)
      set(dat, i=na, j=col, value=NA)
    }

    print('Ordering data')
    order <- c(d$target, setdiff(names(dat), d$target))
    setorderv(dat, order)

    print('Adding ID column')
    set(dat, j='id', value=1L:nrow(dat))

    print('Ordering columns')
    setcolorder(dat, c('id', order))

    print('Saving File')
    write.csv(dat, row.names=FALSE, file=out)
  }

  out <- list(
    dataset_name=paste0("http://test:WtiDBBJWkKjZZwU7@52.4.14.137/datasets/test_data/", out_name),
    metric=d$metric,
    target=d$target
  )

  if(nrow(dat) >= 100000){
    print('Adding big data flag')
    out$worker_size <- '>30'
  }

  return(out)
})

#Save clean and dirty datasets in one file
model_yaml <- as.yaml(c(class_models, reg_models, dirty_datasets))
#cat(model_yaml)
if(overwrite){
  cat(model_yaml, file=paste0(
    '~/workspace/DataRobot/tests/ModelingMachine/',
    'sim_data_num_char_text_miss_up_to_5M.yaml'
  ))
}
