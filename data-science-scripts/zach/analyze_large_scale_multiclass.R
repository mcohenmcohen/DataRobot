######################################################
# Setup
######################################################

stop()
rm(list=ls(all=T))
gc(reset=T)
library(pbapply)
library(data.table)
library(bit64)
library(ggplot2)
library(Hmisc)
library(jsonlite)
library(reshape2)
library(stringi)
library(ggplot2)
library(ggthemes)

######################################################
# Download data
######################################################

mbtest_ids <- c(
  #'5b7f08f17347c90025bc04ae'  # Current with preds
  '5b7578997347c9002bfaefa8'  # Jett's extended multiclass
)

prefix = 'http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests='
suffix = '&max_sample_size_only=false'

urls <- paste0(prefix, mbtest_ids, suffix)
dat_raw <- pblapply(urls, fread)

######################################################
# Convert possible int64s to numeric
######################################################

dat <- copy(dat_raw)

clean_data <- function(x){
  x[,Max_RAM_GB := as.numeric(Max_RAM / 1e9)]
  x[,Total_Time_P1_Hours := as.numeric(Total_Time_P1 / 3600)]
  x[,size_GB := as.numeric(size / 1e9)]
  x[,dataset_size_GB := as.numeric(dataset_size / 1e9)]
  x[,max_vertex_storage_size_P1_MB := as.numeric(max_vertex_storage_size_P1 / 1e6)]
  return(x)
}

dat <- lapply(dat, clean_data)

######################################################
# Combine data
######################################################

get_names <- function(x){
  not_int64 <- sapply(x,  class) != 'integer64'
  names(x)[not_int64]
}

names_all <- Reduce(intersect, lapply(dat, get_names))
stopifnot('Metablueprint' %in% names_all)

dat <- lapply(dat, function(x) x[,names_all,with=F])

dat <- rbindlist(dat, use.names=T)

######################################################
# Add some vars
######################################################

dat[,dataset_bin := cut(dataset_size_GB, unique(c(0, 1.5, 2.5, 5, ceiling(max(dataset_size_GB)))), ordered_result=T, include.lowest=T)]
dat[,sample_round := Sample_Pct]
dat[sample_round=='--', sample_round := '0']
dat[,sample_round := round(as.numeric(sample_round))]

######################################################
# Summary
######################################################
summary(dat[Y_Type == 'Multiclass',list(max_vertex_storage_size_P1_MB)])
dat[!is.na(max_vertex_storage_size_P1_MB) & Y_Type == 'Multiclass',sum(max_vertex_storage_size_P1_MB<=750)/.N]

dat[which.max(max_vertex_storage_size_P1_MB),list(main_task, Filename)]
dat[max_vertex_storage_size_P1_MB>750,table(main_task)]
