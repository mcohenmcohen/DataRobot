# Setup
stop()
rm(list=ls(all=T))
gc(reset=T)
library(data.table)
library(compiler)
library(mbest)
library(kit)
library(ggplot2)
library(ggthemes)

# Load data
files_raw <- c(
  '/Users/zachary/Downloads/test_modules_data_2021_10_06-2021_11_03.csv',
  '/Users/zachary/Downloads/test_modules_data_2021_11_03-2021_11_06.csv'
)
dat_raw <- lapply(files_raw, fread)

# Copy raw data
rm(dat)
gc(reset=T)
dat <- copy(dat_raw)

# Assume the last day in each set is partial
dat <- lapply(dat, function(x){
  max_date <- x[,max(date)]
  return(x[date < max_date,])
})
dat <- rbindlist(dat, fill=T, use.names=T)

# Assume the first day we have is partial
dat <- dat[date > as.IDate('2021-10-06'),]

# Key data
keys <- c('job_name', 'module', 'date')
setkeyv(dat, keys)

# Plots
plot_test <- function(test=NULL, max_size = 1e5, seed=42){
  plot_dat <- dat
  if(!is.null(test)){
    plot_dat <- plot_dat[test,]
  }
  if(nrow(plot_dat) > max_size){
    set.seed(seed)
    plot_dat <- plot_dat[sample(.N, max_size),]
  }
  plt <- ggplot(plot_dat, aes(x=date, y=execution_time)) +
    geom_point() +
    geom_smooth() +
    theme_tufte()
  if(plot_dat[,length(funique(job_name))] < 10){
    if(plot_dat[,length(funique(module))] < 10){
      plt <- plt + facet_wrap(~ job_name + module, scales='free')
    } else{
      plt <- plt + facet_wrap(~ job_name, scales='free')
    }
  }
  print(plt)
}
plot_test('paralleled_integration_executor', 'MLOps/mmm/tests/backend/integration/public_api/deployments/test_external_feature_post.py')
