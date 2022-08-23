rm(list=ls(all=TRUE))
gc(reset=TRUE)
library(bit64)
library(data.table)
x <- fread('~/workspace/att_drinput_07052016.csv')
x <- x[is.finite(buyer),]
unique <- sapply(x, function(a) length(unique(a)))
for(c in names(unique)[unique<=1]){
  set(x, j=c, value=NULL)
}
nums_cols <- names(x)[sapply(x, is.numeric)]
nums_cols <- setdiff(nums_cols, 'buyer')
nums <- x[,nums_cols,with=FALSE]
meds <- sapply(nums, median, na.rm=TRUE)
for(c in names(nums)){
  set(nums, i=which(is.na(nums[[c]])), j=c, value=meds[[c]])
}
anyNA(nums)
unique <- sapply(nums, function(a) length(unique(a)))
for(c in names(nums)[unique<=1]){
  set(nums, j=c, value=NULL)
}

fix_64 <- function(name){
  x <- nums[[name]]
  x <- x - mean(x)
  x  <- x / sd(x)
  set(nums, j=name, value=x)
}
is64 <- names(nums)[sapply(nums, is.integer64)]
for(c in is64){
  fix_64(c)
}

pca <- prcomp(nums, retx=TRUE, center=TRUE, scale=TRUE)
pcax <- data.table(pca$x)
for(c in nums_cols){
  set(x, j=c, value=NULL)
}
x <- data.table(x, pcax)
write.csv(x, '~/workspace/att_drinput_pca.csv', row.names=FALSE)
