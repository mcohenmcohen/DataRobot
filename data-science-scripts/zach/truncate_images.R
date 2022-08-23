# Setup
library(data.table)
library(glmnet)
library(irlba)

# Load Data
x_raw <- fread('http://s3.amazonaws.com/datarobot_public_datasets/images/findleak_full_train_chest_xray.csv')

# Spit x/y
y <- x_raw[,1]
X <- x_raw[,2:ncol(x_raw)]

# Chunk by columns
cols <- split(1:ncol(X), 1:10)
sapply(cols, length)
for (i in seq_along(cols)){
  idx <- cols[[i]]
  out <- data.table(
    y,
    X[,idx,with=F]
  )
  outname <- paste0('~/datasets/findleak_train_chest_xray_chunk', i, '.csv')
  fwrite(out, outname)
}

# PCA
gc(reset=T)
x_pca <- as.matrix(x_raw[,2:ncol(x_raw)])
rm(y, X, cols, out, outname)
gc(reset=T)
set.seed(42)
pca <- prcomp_irlba(x_raw[,2:ncol(x_raw)], n=2500, retx=T)
out <- data.table(
  class=y_glmnet,
  pca$x,
  verbose=T
)
fwrite(out,'~/datasets/findleak_train_chest_xray_pca.csv')

# Glment
y_glmnet <- x_raw[,as.integer(class == 'PNEUMONIA')]
mod <- cv.glmnet(x=pca$x, y=y_glmnet, alpha=0, family='binomial', type.measure = 'auc')
plot(mod)
