library(data.table)
library(readr)
x = fread("https://s3.amazonaws.com/datarobot_public_datasets/gsod_1929_2009_10GB_in_mem_10GB.csv")

b = copy(x)
b[,MeanTemp := as.numeric(MeanTemp)]
b[,DewPoint := as.numeric(DewPoint)]
b[,SLPressure := as.numeric(SLPressure)]
b[,STPressure := as.numeric(STPressure)]
b[,MeanVis := as.numeric(MeanVis)]
b[,MaxWind := as.numeric(MaxWind)]
b[,MeanWind := as.numeric(MeanWind)]
b[,Gust := as.numeric(Gust)]
b[,MaxTemp := as.numeric(MaxTemp)]
b[,MinTemp := as.numeric(MinTemp)]
b[,Precipitation := as.numeric(Precipitation)]
b[,SnowDepth := as.numeric(SnowDepth)]
b = b[!is.na(MaxTemp),]

nums = sapply(b, is.numeric)
keep = names(nums)[nums]
a = b[,keep,with=F]
for(j in names(a)){
  v = a[[j]]
  i = which(is.na(v))
  set(a, i, j, value=median(v, na.rm=TRUE))
}
first = 'MaxTemp'
setcolorder(a, c(first, setdiff(names(a), first)))
set.seed(42)
a = a[sample(1:.N, 1e6),]
write_csv(a, '~/datasets/GSOD_num_1e6.csv')
