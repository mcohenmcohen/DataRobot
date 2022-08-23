library(data.table)
library(Matrix)
library(irlba)
library(pbapply)
library(text2vec)
dat <- fread('~/datasets/train_combined_date.csv')
t_id = dat[['TransactionID']]
dat[,`Unnamed: 0` := NULL]
dat[,TransactionID := NULL]
dat[,isFraud := NULL]
dat[,TransactionDateTime := NULL]
gc(reset=T)

for(var in names(dat)){
  x <- dat[[var]]
  if(is.numeric(x)){
    x[is.na(x)] <- median(x, na.rm = T)
    x <- scale(x, center = T, scale = T)
    stopifnot(all(is.finite(x)))
  } else{
    x <- factor(x)
    x <- addNA(x, ifany = T)
  }
  set(dat, j=var, value=x)
}
gc(reset=T)
X <- model.matrix(~0+., dat)

pca_comps <- prcomp_irlba(X, retx=T, center=F, scale=F, n=100, verbose=T)

pca_x <- pca_comps$x
gc(reset=T)

center_point = colMeans(pca_x)

distances = dist2(pca_x,matrix(center_point, nrow=1), method='euclidean')

t_id[which.max(distances)]
dat[which.max(distances),]
