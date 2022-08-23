
########################################################
# Libraries
########################################################

library(data.table)
library(mongolite)
library(jsonlite)
library(ggplot2)
rm(list=ls(all=T))
gc(reset=T)

#https://ropensci.org/blog/blog/2017/03/10/mongolite
#https://jeroen.github.io/mongolite/

########################################################
# Useful functions
########################################################

task_info_table <- function(res, important_cols_only=TRUE){
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
  out <- t(out)
  
  return(out)
}

pull_lid <- function(lid, filter, host){
  library(mongolite)
  con <- mongolite::mongo(
    db = 'MMApp',
    collection = 'leaderboard',
    url=paste0('mongodb://', host), verbose=F)
  res = con$find(
    query = paste0('{"_id": {"$oid":"', lid, '"}}'),
    fields = paste0('{', filter, ', "_id": false}'),
    limit=1)
  rm(con)
  sink <- gc()
  return(res)
}

blueprint_info <- function(lid, host){
  res <- pull_lid(lid, '"task_info":true', host)
  return(task_info_table(res))
}

blueprint_json <- function(lid, host="10.20.53.43"){
  library(jsonlite)
  res <- pull_lid(lid, '"blueprint":true', host)
  res <- toJSON(res, pretty=TRUE)
  return(res)
}

get_both <- function(lid, host="10.20.53.43"){
  print(blueprint_json(lid, host))
  print(blueprint_info(lid, host))
}

########################################################
# 100GB
########################################################

