library(data.table)
dat = fread("~/Downloads/advanced_export (1).csv")

dat[,list(Filename, model_family, MASE_H)]
dat[,MASE_H := as.numeric(MASE_H)]
dat[,list(mean(MASE_H, na.rm=T)), by=main_task][order(V1),]
