########################################################
# Libraries
########################################################

rm(list=ls(all=T))
gc(reset=T)

library(data.table)
library(mongolite)
library(jsonlite)
library(pbapply)
library(ggplot2)
library(ggthemes)

host <- "shrink-mongo-0.infra.ent.datarobot.com"
port <- 27017

########################################################
# Functions to load and format data
########################################################

task_info_table <- function(res, transpose=T){
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
  for(n in c(
    'fit.max.RAM', 'fit.avg.RAM', 'transform.max.RAM', 'fit.total.RAM',
    'transform.avg.RAM', 'transform.total.RAM', 'fit.CPU.time',
    'fit.clock.time', 'transform.clock.time', 'transform.CPU.time',
    'transform.sys.time', 'fit.sys.time')){
    if(! n %in% names(out))
      set(out, j=n, value=as.numeric(NA))
  }

  out[, fit.max.RAM := fit.max.RAM / 1024^3]
  out[, transform.max.RAM := transform.max.RAM / 1024^3]

  out[, fit.clock.time := fit.clock.time / 3600]
  out[, transform.clock.time := transform.clock.time / 3600]

  setnames(out, gsub(".RAM", ".RAM.GB", names(out), fixed=TRUE))
  setnames(out, gsub(".time", ".time.hours", names(out), fixed=TRUE))

  #Select cols
  select <- c(
    'task_name',
    'fit.max.RAM.GB',
    'transform.max.RAM.GB',
    'fit.clock.time.hours',
    'transform.clock.time.hours'
  )

  out <- out[,select,with=F]

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

pull_id <- function(lid, filter=NULL, collection, id_name="_id", limit=1){
  library(mongolite)
  con <- mongolite::mongo(
    db = 'MMApp',
    collection = collection,
    url=paste0('mongodb://', host, ":", port), verbose=F)
  if(is.null(filter)){
    res = con$find(
      query = paste0('{"', id_name, '": {"$oid":"', lid, '"}}'),
      limit=1)
  } else {
    res = con$find(
      query = paste0('{"', id_name, '": {"$oid":"', lid, '"}}'),
      fields = ifelse(id_name=="_id", paste0('{', filter, ', "',id_name,'": false}'),
                      paste0('{', filter, '}')),
      limit=limit)
  }
  rm(con)
  sink <- gc()
  return(res)
}

pull_lid <- function(lid, filter=NULL){
  return(pull_id(lid, filter, collection='leaderboard'))
}

pull_lid_by_pid <- function(pid, filter=NULL){
  return(pull_id(pid, filter, collection='leaderboard', id_name="pid", limit = 0))
}

blueprint_json <- function(lid, pretty=TRUE){
  library(jsonlite)
  res <- pull_lid(lid, '"blueprint":1')
  res <- toJSON(res, pretty=pretty)
  return(res)
}

lid_csv_line <- function(lid){
  res = pull_lid(lid, '"test.Gini Norm":1, "task_info":1')
  info_table = task_info_table(res)
  run_time = sum(c(info_table["fit.clock.time.hours",], info_table["transform.clock.time.hours",]), na.rm=T)
  max_ram = max(c(info_table["transform.max.RAM.GB",], info_table["fit.max.RAM.GB",]), na.rm=T)
  gini_norm = res[["test"]][["Gini Norm"]][[1]]
  blueprint = as.character(blueprint_json(lid, FALSE))

  dframe = data.table(lid, run_time, max_ram, gini_norm, blueprint)
  return(dframe)
}

lid_info_summary<- function(project_id){

  lids = pull_lid_by_pid(project_id, '"_id":1')

  dframes <- pblapply(lids[["_id"]], function(v){
    #print(v)
    out <- tryCatch(lid_csv_line(v), error=function(e) NULL)
    return(out)
  })

  summary_df <- rbindlist(dframes, fill=TRUE, use.names=TRUE)
  return(summary_df)
}

########################################################
# Pull data (this takes a couple minutes)
########################################################

data_raw_dss <- lid_info_summary("5dc548b472999a004d7e5813")
data_raw_scalable <- lid_info_summary("5dc543cf3887ef005c503d81")

########################################################
# Analyze data
########################################################

data_dss = copy(data_raw_dss)
data_scalable = copy(data_raw_scalable)

summary(data_dss)
summary(data_scalable)

both <- merge(data_dss, data_scalable, by='blueprint', all=T, suffixes=c('_dss', '_scalable'))
both[,gini_diff := gini_norm_scalable - gini_norm_dss]
both[,gini_pct_diff := gini_diff / gini_norm_dss]
summary(both[,list(gini_norm_dss, gini_norm_scalable, gini_diff, gini_pct_diff)])

ggplot(both, aes(x=gini_norm_dss, y=gini_norm_scalable)) +
  geom_point() +
  geom_abline(slope=1, intercept = 0) +
  theme_tufte()

# gini_norm_dss    gini_norm_scalable   gini_diff        gini_pct_diff
# Min.   :0.0000   Min.   :0.0000     Min.   :-0.09314   Min.   :-0.41900
# 1st Qu.:0.3082   1st Qu.:0.2794     1st Qu.:-0.03356   1st Qu.:-0.10489
# Median :0.3360   Median :0.3192     Median :-0.00690   Median :-0.03218
# Mean   :0.3032   Mean   :0.2929     Mean   :-0.01038   Mean   :-0.00932
# 3rd Qu.:0.3750   3rd Qu.:0.3533     3rd Qu.: 0.01428   3rd Qu.: 0.05445
# Max.   :0.4016   Max.   :0.4141     Max.   : 0.07758   Max.   : 1.23517
# NA's   :5        NA's   :7          NA's   :12         NA's   :13
