library(readr)
library(data.table)
library(jsonlite)
library(pbapply)
dat <- read_csv('/Users/zachary/Downloads/Multilabel Target Column 93 Labels (1).csv')
dat <- data.table(dat)
dat[,label := pbsapply(PathTests, fromJSON)]
all_labels = dat[,sort(table(unlist(label)))]
stopifnot(length(all_labels) <= 100)


dat[which(sapply(label, function(x) "" %in% x)),]

dat[,which(sapply(label, function(x) "" %in% x))]
