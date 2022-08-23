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

old_mbtest_ids <- c(
  '5ce5630d7347c9002707da6a'
  #'5d6fee2b7347c90024eff99e'
  )
new_mbtest_ids <- c(
  '5d6fc24c7347c9002cd2e91a'
  #'5d6fc2937347c900290e0bc8'
)

test <- 'Gaussian_Process_Test'

prefix = 'http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests='
suffix = '&max_sample_size_only=false'

old_mbtest_urls <- paste0(prefix, old_mbtest_ids, suffix)
new_mbtest_urls <- paste0(prefix, new_mbtest_ids, suffix)

dat_old_raw <- pblapply(old_mbtest_urls, fread)
dat_new_raw <- pblapply(new_mbtest_urls, fread)

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
  return(x)
}

dat_old <- lapply(dat_old, clean_data)
dat_new <- lapply(dat_new, clean_data)

stopifnot(all(sapply(dat_old, function(x) 'Max_RAM_GB' %in% names(x))))
stopifnot(all(sapply(dat_new, function(x) 'Max_RAM_GB' %in% names(x))))

######################################################
# Combine data
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

dat_old <- dat_old[!grepl("Z", dat_old$training_length),]
dat_new <- dat_new[!grepl("Z", dat_new$training_length),]

dat_old[,run := 'master']
dat_new[,run := 'GPRs']

stopifnot(dat_old[,all(Metablueprint=='Metablueprint v12.0.03-so' | Metablueprint=='')])
stopifnot(dat_new[,all(Metablueprint=='TestGPRs v2')])

dat <- rbindlist(list(dat_old, dat_new), use.names=T)

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

# Exlcude Prime
dat <- dat[is_prime == FALSE,]

# Exclide auto-tuned word gram models from the GPR tests
dat <- dat[!(run=='GPRs' & main_task=='WNGER2'),] 

######################################################
# Parse BPs
######################################################

dat[grepl('vary_length_scaling=1', Blueprint), run := 'vary_length_scaling_1']
dat[grepl('vary_length_scaling=0', Blueprint), run := 'vary_length_scaling_0']

######################################################
# Organize data
######################################################

res <- copy(dat)

res <- res[,list(
  Max_RAM_GB = max(Max_RAM_GB, na.rm=T),
  Total_Time_P1_Hours = max(Total_Time_P1_Hours, na.rm=T),
  RMSE_V = min(RMSE_P1, na.rm=T),
  RMSE_H = min(RMSE_H, na.rm=T)
), by=c('run', 'Filename', 'Y_Type', 'sample_round')]

measures = c('Max_RAM_GB', 'Total_Time_P1_Hours', 'RMSE_V', 'RMSE_H')
for(v in measures){
  tmp = sort(unique(res[[v]]))
  wont_convert = !is.finite(as.numeric(tmp))
  if(any(wont_convert)){
    print(tmp[wont_convert])
  }
  set(res, j=v, value=as.numeric(res[[v]]))
}
res = melt.data.table(res, measure.vars=intersect(names(res), measures))
res = dcast.data.table(res, Filename + Y_Type + sample_round + variable ~ run, value.var='value')

set(res, j='diff_vary_length_scaling_1', value = res[['vary_length_scaling_1']] - res[['master']])
set(res, j='diff_vary_length_scaling_0', value = res[['vary_length_scaling_0']] - res[['master']])

######################################################
# Plot of results
######################################################

plot_dat <- copy(res)
plot_dat <- plot_dat[variable %in% c('Max_RAM_GB', 'Total_Time_P1_Hours', 'RMSE_V', 'RMSE_H'),]
plot_dat <- plot_dat[!is.na(vary_length_scaling_0),]
plot_dat <- plot_dat[!is.na(vary_length_scaling_1),]
plot_dat <- plot_dat[!is.na(master),]

ggplot(plot_dat, aes(x=master)) + 
  geom_point(aes(y=vary_length_scaling_1, color='vary_length_scaling=1')) + 
  geom_point(aes(y=vary_length_scaling_0, color='vary_length_scaling=0')) + 
  geom_abline(slope=1, intercept=0) + 
  facet_wrap(~variable, scales='free') + 
  theme_bw() + theme_tufte() + ggtitle('master vs GPR results')

######################################################
# Lookit issues
######################################################

# Runtime diffs
res[sample_round==64 & diff_vary_length_scaling_1 > 0.1 & variable %in% c('Total_Time_P1_Hours'),][order(diff_vary_length_scaling_1),]
