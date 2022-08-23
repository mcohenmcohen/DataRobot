library(data.table)

raw <- fread('~/workspace/data-science-scripts/zach/no_transformation.csv', header=T)
std <- fread('~/workspace/data-science-scripts/zach/standardization.csv', header=T)
rdt <- fread('~/workspace/data-science-scripts/zach/ridit.csv', header=T)

raw[,V1 := NULL]
std[,V1 := NULL]
rdt[,V1 := NULL]

summary(unlist(raw))
summary(unlist(std))
summary(unlist(rdt))
