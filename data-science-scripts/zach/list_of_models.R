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
# https://datarobot.atlassian.net/wiki/spaces/QA/pages/111691866/Release+MBTests

mbtest_ids <- c(
  '5c545d417347c900255bf2db',  # Nightly combined
  '5c51ba527347c9002ccca829',  # Current with preds
  '5c40da04a2c902000142a185',  # 4.5 release
  '5c4243e5e3cb9e0001416923',  # 4.5 release
  '5c4b4ed8ba08880001b81d9b',  # 4.5 release
  '5c40e4b330cf35000175821c',  # 4.5 release
  '5c4243c42ce6130001323a60',  # 4.5 release
  '5c461f85970b8900011b19d4'  # 4.5 release
)

prefix = 'http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests='
suffix = '&max_sample_size_only=false'

mbtest_urls <- paste0(prefix, mbtest_ids, suffix)
dat_raw <- pblapply(mbtest_urls, fread)

######################################################
# Convert possible int64s to numeric
######################################################

dat <- copy(dat_raw)

clean_data <- function(x){
  x[,Max_RAM_GB := as.numeric(Max_RAM / 1e9)]
  x[,Total_Time_P1_Hours := as.numeric(Total_Time_P1 / 3600)]
  x[,size_GB := as.numeric(size / 1e9)]
  x[,dataset_size_GB := as.numeric(dataset_size / 1e9)]
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
# Counts
######################################################

dat[,length(unique(Blueprint))]  # 8089
dat[,length(unique(Filename))]  # 456
dat[,length(unique(main_task))]  # 120

tasks <- dat[,unique(main_task)]
tasks <- gsub('C$', '', tasks)
tasks <- gsub('R$', '', tasks)
tasks <- unique(tasks)
length(tasks)  # 101
