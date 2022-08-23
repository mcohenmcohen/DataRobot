######################################################
# Setup
######################################################

stop()
library(data.table)
library(bit64)
library(ggplot2)
library(Hmisc)
library(pbapply)

######################################################
# Download data
######################################################
#https://datarobot.atlassian.net/browse/QA-2435

wrap_test <- function(...){
  x <- as.data.table(t(c(...)))
  setnames(x, c('version', 'platform', 'datasets', 'id'))
  return(x)
}
mbtests <- list(
  wrap_test(3.1, 'docker', 'small', '59404c19422e2f10da154169'),
  wrap_test(3.1, 'YARN', 'small', '59404db027f1752af00d58dc'),
  wrap_test(3.1, 'docker', 'large', '59404e2b27f1752bee98ad83'),
  wrap_test(3.0, 'docker', 'small', '59405479eb04ec1141d7ad0f')
)

mbtests <- rbindlist(mbtests)

make_link <- function(x){
  paste0('http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests=', x, '&max_sample_size_only=false')
}
dat_list <- pblapply(mbtests$id, function(x){
  out <- fread(make_link(x))
  out[,mbtest_id := x]
  return(out)
})

######################################################
# Organize data
######################################################

dat <- rbindlist(dat_list, fill=T)
dat <- merge(dat, mbtests, all.x=T, by.x='mbtest_id', by.y='id')
dat[,table(version, useNA = 'ifany')]

dat[,Max_RAM_GB := as.numeric(Max_RAM * 1e-9)]
dat[,Total_Time_P1_Hours := Total_Time_P1/3600]
dat[,size_GB := size * 1e-9]
dat[,dataset_size_GB := as.numeric(dataset_size * 1e-9)]

dat[,dataset_bin := cut(dataset_size_GB, c(0, .5, 1.5, 2.5, 5, ceiling(max(dataset_size_GB))), ordered_result=T, include.lowest=T)]
dat[,table(dataset_bin, useNA = 'ifany')]
######################################################
# Tables
######################################################

dat[version == 3.1, summary(`Gini Norm_P1`)]
dat[version == 3.1, summary(Max_RAM_GB)]
dat[version == 3.1, summary(Total_Time_P1_Hours)]
dat[version == 3.1, summary(size_GB)]
dat[version == 3.1, summary(dataset_size_GB)]

######################################################
# Bar chars
######################################################

#RAM
ggplot(dat, aes(x=dataset_bin, fill=version, y=Max_RAM_GB)) +
  stat_summary(fun.y = mean, geom = "bar", position=position_dodge()) +
  stat_summary(fun.data = mean_cl_boot, geom = "errorbar", width=0.2, position=position_dodge(width=.9)) +
  theme_bw() +
  ggtitle('Max RAM usage by Dataset Size') +
  xlab('Dataset Size (GB) + 95% CI') +
  ylab('Max RAM Usage')

#Runtime
ggplot(dat, aes(x=dataset_bin, fill=version, y=Total_Time_P1_Hours)) +
  stat_summary(fun.y = mean, geom = "bar", position=position_dodge()) +
  stat_summary(fun.data = mean_cl_boot, geom = "errorbar", width=0.2, position=position_dodge(width=.9)) +
  theme_bw() +
  ggtitle('Max RAM usage by Dataset Size') +
  xlab('Runtime (Hours) + 95% CI') +
  ylab('Max RAM Usage')

#Accuracy
ggplot(dat, aes(x=dataset_bin, fill=version, y=`Gini Norm_P1`)) +
  stat_summary(fun.y = mean, geom = "bar", position=position_dodge()) +
  stat_summary(fun.data = mean_cl_boot, geom = "errorbar", width=0.2, position=position_dodge(width=.9)) +
  theme_bw() +
  ggtitle('Max RAM usage by Dataset Size') +
  xlab('Accuracy (Gini Norm) + 95% CI') +
  ylab('Max RAM Usage')

######################################################
# Scatter plots
######################################################

plot_dat <- dat[,list(Filename, Blueprint, Sample_Pct, version, `Gini Norm_P1`, Max_RAM_GB, Total_Time_P1_Hours, dataset_size_GB)]
plot_dat <- melt.data.table(plot_dat)
plot_dat <- dcast(plot_dat, Filename + Blueprint + Sample_Pct ~ variable + version)
