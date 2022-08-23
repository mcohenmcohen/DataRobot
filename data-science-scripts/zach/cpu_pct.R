#Load Data
library('shrinkR')
library('data.table')
library('reshape2')
library('ggplot2')
library('ggrepel')
library('scales')
library('jsonlite')
library('pbapply')
library('stringi')
library('readr')

#Load base MBtest and new run
#Sys.sleep(43200)
# suppressWarnings(dat <- getLeaderboard('5812156ffc5c7411755c04f8'))
# setnames(dat, make.names(names(dat), unique=TRUE))
# dat[,length(unique(Filename))]

#x <- fread('~/Downloads/nov-cpu-info.csv')
x <- fread('~/Downloads/4-months.csv')
x[,model_type := gsub('\\(.*?\\)', '', model_type)]
x[,model_type := pbsapply(stri_split_fixed(model_type, ' - '), '[', 1)]
x[,model_type := stri_trim_both(model_type)]
x <- x[!is.na(cpu_percent),]
x <- x[,list(
  mean_cpu_percent = mean(cpu_percent) / 100,
  min_cpu_percent = min(cpu_percent) / 100,
  max_cpu_percent = max(cpu_percent) / 100,
  n = .N
), by='model_type']
setorder(x, -n)
head(x, 100)
write_csv(x, '~/datasets/cpu_pct_4.csv')
