rm(list=ls(all=T))
gc(reset=TRUE)

# Random sparse matrix
library(Matrix)
set.seed(42)
n_docs <- 1e4
m <- rsparsematrix(n_docs, n_docs, density=0.10, symmetric=T, rand.x=runif)

# Find top5 most similar docs
library(data.table)
dt <- data.table(summary(m))
setkeyv(dt, c('i', 'x'))  # i is row, j is col
dt[,rank := .N:1, by='i']
dt <- dt[rank <= 5,]
setkeyv(dt, c('i', 'rank'))
head(dt, 20)
