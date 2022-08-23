stop()
rm(list=ls(all=T))
gc(reset=T)
library(yaml)
library(data.table)
library(readr)
library(httr)
library(pbapply)
library(stringi)
library(ggplot2)
library(scales)
library(viridis)
library(ggthemes)
#https://datarobot.atlassian.net/browse/MBP-1545

#dat_raw <- fread('http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests=59cd3ebb8b6eea00016b1fe3&max_sample_size_only=false')

ids <- c(
  '59ed2201cbd1920001b2ab7d',
  '59ef62752c775c0001631291',
  '59f1003fb2be610001b252c4',
  '59f7339d262ac900017f6571',
  '5a04977610a55900018a5719',
  '5a09b4ab107db90001928534'
)

dat_raw <- pblapply(ids, function(x){
  url <- paste0(
    'http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests=',
    x,
    '&max_sample_size_only=false')
  fread(url)
})

TABLE_FILE <- '~/workspace/data-science-scripts/zach/multiclas_yaml_class_count.csv'
class_table <- fread(TABLE_FILE)

dat <- rbindlist(dat_raw)
stopifnot(dat[,length(unique(Filename))] == 260)

dat[,Max_RAM_GB := as.numeric(Max_RAM) * 1e-9]
dat[,Total_Time_P1_Hours := Total_Time_P1/3600]
dat[,size_GB := size * 1e-9]

dat[,Sample_Pct := as.numeric(Sample_Pct)]
dat[,table(Sample_Pct)]

x = dat[,list(main_task, Filename, Sample_Size, Sample_Pct, LogLoss_P1, Total_Time_P1_Hours, Max_RAM_GB, size_GB)]
summary(x)
x[Total_Time_P1_Hours > 2,][order(Total_Time_P1_Hours),]

agg <- dat[
  !is.na(LogLoss_P1) & Sample_Pct <= 64.3
  ,list(
  logloss = min(LogLoss_P1),
  time = sum(Total_Time_P1_Hours),
  Sample_Pct = max(Sample_Pct),
  Max_RAM_GB = max(Max_RAM_GB),
  Sample_Size = Sample_Size[1],
  x_cols = x_cols[1],
  x_cat = x_cat[1],
  x_txt = x_txt[1],
  x_dates = x_dates[1],
  x_numeric = x_numeric[1]
  ), by='Filename']

agg[,Filename := gsub('_80.csv', '', Filename, fixed=T)]
agg[,cells := Sample_Size * x_cat]

names <- stri_split_fixed(class_table$dataset_name, pattern='/')
names <- sapply(names, function(x) rev(x)[1])
class_table[,dataset_name := names]
class_table[,dataset_name := gsub('openML/datasets/', '', dataset_name, fixed=T)]
class_table[,dataset_name := gsub('openML/large/', '', dataset_name, fixed=T)]
class_table[,dataset_name := gsub('text/datasets/', '', dataset_name, fixed=T)]
class_table[,dataset_name := gsub('openML/large/', '', dataset_name, fixed=T)]
class_table[,dataset_name := gsub('.csv', '', dataset_name)]
class_table <- unique(class_table)
class_table[order(nchar(dataset_name)),]

agg <- merge(agg, class_table, by.x='Filename', by.y='dataset_name', all.x=T)

agg[,list(.N), by=list(is.na(classes))]
agg[is.na(classes),][order(nchar(Filename)),]

agg[logloss > 1,][order(logloss),]
agg[time > 10,][order(time),]

ggplot(agg, aes(x=classes, y=logloss)) + geom_point() + geom_smooth() +
  theme_tufte(base_family="Helvetica")

ggplot(agg, aes(x=classes, y=time)) + geom_point() + geom_smooth() +
  theme_tufte(base_family="Helvetica")

ggplot(agg, aes(x=classes, y=Max_RAM_GB)) + geom_point() + geom_smooth() +
  theme_tufte(base_family="Helvetica")

ggplot(agg, aes(x=factor(classes), y=logloss)) + geom_boxplot() +
  theme_tufte(base_family="Helvetica")

ggplot(agg, aes(x=factor(classes), y=time)) + geom_boxplot() +
  theme_tufte(base_family="Helvetica") + scale_y_log10()

ggplot(agg, aes(x=Sample_Size, y=time)) + geom_point() + geom_smooth() +
  theme_tufte(base_family="Helvetica")
agg[time>10,min(Sample_Size)]

ggplot(agg, aes(x=Sample_Size, y=Max_RAM_GB)) + geom_point() + geom_smooth() +
  theme_tufte(base_family="Helvetica")

ggplot(agg, aes(x=Sample_Size, y=logloss)) + geom_point() + geom_smooth() +
  theme_tufte(base_family="Helvetica") + scale_x_log10()

ggplot(agg, aes(x=I(Sample_Size * classes), y=logloss)) + geom_point() + geom_smooth() +
  theme_tufte(base_family="Helvetica") + scale_x_log10()

ggplot(agg, aes(x=I(Sample_Size * classes), y=time)) + geom_point() + geom_smooth() +
  theme_tufte(base_family="Helvetica") + scale_x_log10() + scale_y_log10()

ggplot(agg, aes(x=Sample_Size, y=time)) + geom_point() + geom_smooth() +
  theme_tufte(base_family="Helvetica") + scale_x_log10()

ggplot(agg, aes(x=x_cols, y=time)) + geom_point() + geom_smooth() +
  theme_tufte(base_family="Helvetica") + scale_x_log10()

ggplot(agg, aes(x=x_cat, y=time)) + geom_point() + geom_smooth() +
  theme_tufte(base_family="Helvetica")

ggplot(agg, aes(x=x_txt, y=time)) + geom_point() + geom_smooth() +
  theme_tufte(base_family="Helvetica")

ggplot(agg, aes(x=x_dates, y=time)) + geom_point() + geom_smooth() +
  theme_tufte(base_family="Helvetica")

ggplot(agg, aes(x=x_numeric, y=time)) + geom_point() + geom_smooth() +
  theme_tufte(base_family="Helvetica")

ggplot(agg, aes(x=cells, y=time)) + geom_point() + geom_smooth() +
  theme_tufte(base_family="Helvetica")
