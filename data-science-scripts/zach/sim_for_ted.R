rm(list=ls(all=T))
gc(reset=T)
library(data.table)

dat <- data.table(t(combn(9, 4)))
dat[,is_seq := (V4 - V3 == 1) & (V3 - V2 == 1) & (V2 - V1 == 1)]
dat[which(is_seq),]
dat[which(!is_seq),]
dat[,sum(is_seq) / .N]


library(combinat)
permn(9, 4)
