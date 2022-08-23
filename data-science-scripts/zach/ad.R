rm(list=ls(all=T))
gc(reset=T)
library(readr)
library(data.table)
library(AnomalyDetection)
data(raw_data)
res = AnomalyDetectionTs(raw_data, max_anoms=0.02, direction='both', plot=TRUE)

out <- copy(raw_data)
out$timestamp <- as.POSIXct(out$timestamp,tz="UTC")
res$anoms$timestamp <- as.POSIXct(res$anoms$timestamp,tz="UTC")

out <- merge(data.table(out), data.table(res$anoms), all.x=T, by='timestamp')
out[is.na(anoms), anoms := 0]
out[anoms==1,]

out[,timestamp := format(timestamp)]
write_csv(out, '~/datasets/twitter_ad.csv')


pred <- fread('~/Downloads/twitter_ad.csv_Isolation_Forest_Anomaly_Detection_(4)_0_Timeseries_Informative_Features_twitter_ad.csv')
pred[,Timestamp := as.POSIXct(Timestamp)]
setnames(pred, tolower(names(pred)))
pred <- merge(out[,list(timestamp=as.POSIXct(timestamp), count, anoms)], pred, by='timestamp')
plot(count~timestamp, pred)
lines(anoms~timestamp, pred)
plot(prediction~timestamp, pred)

