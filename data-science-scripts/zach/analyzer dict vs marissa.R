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

# Newer set of marisa Mbtests
old_mbtest_ids <- c(
  '5c9a99af7347c9002493f763',  # FM Yaml
  '5c9a94d67347c900273dbf13',  # Current with preds Yaml
  '5c9a99a27347c900250bcfaa',  # Cosine sim
  '5c9a99cd7347c900273dc103'  # Single column text
)

# Run 9
new_mbtest_ids <- c(
  '5cab6f397347c90024b734ee',  # FM Yaml
  '5cab6ecb7347c90024b73314',  # Current with preds Yaml
  '5cab6f037347c9002902a7f1',  # Cosine sim
  '5cab6fd07347c90026da19b8'  # Single column text
)

# Name 'em
testnames <- c(
  'Factorization Machines', 'Current With Preds',
  'Cosine Sim', 'Single Text')
names(old_mbtest_ids) <- testnames
names(new_mbtest_ids) <- testnames
all_tests <- c(old_mbtest_ids, new_mbtest_ids)

prefix = 'http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests='
suffix = '&max_sample_size_only=false'

# Read and name
read_and_name <- function(id){
  url <- paste0(prefix, id, suffix)
  out <- fread(url)
  out[,mbtest_id := id]
  out[,mbtest_name := names(all_tests[id == all_tests])]
  return(out)
}
dat_old_raw <- pblapply(old_mbtest_ids, read_and_name)
dat_new_raw <- pblapply(new_mbtest_ids, read_and_name)

######################################################
# Convert possible int64s to numeric
######################################################

dat_old <- copy(dat_old_raw)
dat_new <- copy(dat_new_raw)

clean_data <- function(x){
  x[,Max_RAM_GB := as.numeric(Max_RAM / 1e9)]
  x[,Total_Time_P1_Hours := as.numeric(Total_Time_P1 / 3600)]
  x[,size_GB := as.numeric(size / 1e9)]
  x[,dataset_size_GB := as.numeric(dataset_size / 1e9)]
  x[,x_prod_2_max_cardinal := NULL]
  return(x)
}

dat_old <- lapply(dat_old, clean_data)
dat_new <- lapply(dat_new, clean_data)

stopifnot(all(sapply(dat_old, function(x) 'Max_RAM_GB' %in% names(x))))
stopifnot(all(sapply(dat_new, function(x) 'Max_RAM_GB' %in% names(x))))

######################################################
# Combine data within each test
######################################################

get_names <- function(x){
  not_int64 <- sapply(x,  class) != 'integer64'
  names(x)[not_int64]
}

names_old <- Reduce(intersect, lapply(dat_old, get_names))
names_new <- Reduce(intersect, lapply(dat_new, get_names))
names_all <- intersect(names_new, names_old)

stopifnot('Metablueprint' %in% names_all)

dat_old <- lapply(dat_old, function(x) x[,names_all,with=F])
dat_new <- lapply(dat_new, function(x) x[,names_all,with=F])

dat_old <- rbindlist(dat_old, use.names=T)
dat_new <- rbindlist(dat_new, use.names=T)

dat_old[,run := 'marisa']
dat_new[,run := 'dict']

stopifnot(dat_old[,all(Metablueprint=='Metablueprint v12.0.03-so')])
stopifnot(dat_new[,all(Metablueprint=='Metablueprint v12.0.03-so')])

######################################################
# Combine data BETWEEN the 2 tests
######################################################

# Combine into 1
dat <- rbindlist(list(dat_old, dat_new), use.names=T)

# Map names to test
filename_to_test_map <- unique(dat[,list(Filename, mbtest_name)])
filename_to_test_map <- filename_to_test_map[!duplicated(Filename),]

######################################################
# Add some vars
######################################################

dat[,dataset_bin := cut(dataset_size_GB, unique(c(0, 1.5, 2.5, 5, ceiling(max(dataset_size_GB)))), ordered_result=T, include.lowest=T)]
dat[,sample_round := Sample_Pct]
dat[sample_round=='--', sample_round := '0']
dat[,sample_round := round(as.numeric(sample_round))]

######################################################
# Exclude some rows
######################################################

# Exclude blenders and primes
dat <- dat[which(!is_blender),]
dat <- dat[which(!is_prime),]

# Only look at datases with text
dat <- dat[dataset_x_txt>0,]

######################################################
# Summarize stats - non multiclass
######################################################

# Find a var
# a=sort(names(dat)); a[grepl('Y_Type', tolower(a))]
res <- copy(dat)
res <- res[!is.na(Max_RAM_GB),]
res <- res[!is.na(Total_Time_P1_Hours),]
res <- res[!is.na(`Gini Norm_H`),]

res <- res[,list(
  Max_RAM_GB = max(Max_RAM_GB),
  Total_Time_P1_Hours = max(Total_Time_P1_Hours),
  Gini_V = max(`Gini Norm_P1`),
  Gini_H = max(`Gini Norm_H`),
  Gini_P = max(`Prediction Gini Norm`),
  MASE_H = min(`MASE_H`),
  MASE_V = min(`MASE_P1`),
  LogLoss_H = min(`LogLoss_H`),
  LogLoss_V = min(`LogLoss_P1`)
), by=c('run', 'Filename', 'Y_Type')]

measures = c(
  'Max_RAM_GB', 'Total_Time_P1_Hours', 'Gini_V', 'Gini_H', 'Gini_P', 'MASE_H', 'MASE_V', 'LogLoss_H', 'LogLoss_V')
for(v in measures){
  tmp = sort(unique(res[[v]]))
  wont_convert = !is.finite(as.numeric(tmp))
  if(any(wont_convert)){
    print(tmp[wont_convert])
  }
  set(res, j=v, value=as.numeric(res[[v]]))
}

res = melt.data.table(res, measure.vars=intersect(names(res), measures))
res = dcast.data.table(res, Filename + Y_Type + variable ~ run, value.var='value')

res[,diff := as.numeric(dict) - as.numeric(marisa)]

# Add test name
N <- nrow(res)
res <- merge(res, filename_to_test_map, all.x=T, by=c('Filename'))
stopifnot(N == nrow(res))

######################################################
# Plot of results - non multiclass
######################################################

plot_vars = c('Max_RAM_GB', 'Total_Time_P1_Hours', 'Gini_V', 'Gini_H')
plotdat <- res[
  variable %in% plot_vars & !is.na(dict) & !is.na(marisa),]
ggplot(plotdat, aes(x=`marisa`, y=`dict`, color=mbtest_name)) + 
  geom_point() + geom_abline(slope=1, intercept=0) + 
  facet_wrap(~variable, scales='free') + 
  theme_bw() + theme_tufte() + ggtitle('dict vs marisa results')

res[dict > marisa & variable=='Max_RAM_GB',]
res[dict > marisa & variable=='Total_Time_P1_Hours',]

res[variable=='Max_RAM_GB' & !is.na(dict) & !is.na(marisa), sum(dict < marisa)/.N]
res[variable=='Total_Time_P1_Hours' & !is.na(dict) & !is.na(marisa), sum(dict < marisa)/.N]
