library(data.table)
library(fpc)
df = fread('~/workspace/data-science-scripts/zach/df.csv', header=T)
setnames(df, c('y', 'x'))
plot(y ~ x , df)

# SVD, k=1 then kmeans
png('~/workspace/data-science-scripts/zach/bad_svd_k.png')
data_preprocessed <- scale(as.matrix(df), center=T, scale=T)
data_preprocessed <- svd(data_preprocessed, nu=1, nv=1)$u
clusters_kmeans_svd <- kmeansruns(data_preprocessed, critout=T, criterion = 'ch', krange=4, iter.max=10, runs=1)
plot(y ~ x , df, col=clusters_kmeans_svd$cluster)
title('SVD k=1, then kmeans')
dev.off()

# SVD, k=2 then kmeans
png('~/workspace/data-science-scripts/zach/good_svd_k.png')
data_preprocessed <- scale(as.matrix(df), center=T, scale=T)
data_preprocessed <- svd(data_preprocessed, nu=2, nv=2)$u
clusters_kmeans_svd <- kmeansruns(data_preprocessed, critout=T, criterion = 'ch', krange=4, iter.max=10, runs=1)
plot(y ~ x , df, col=clusters_kmeans_svd$cluster)
title('SVD k=2, then kmeans')
dev.off()

system('open ~/workspace/data-science-scripts/zach/bad_svd_k.png')
system('open ~/workspace/data-science-scripts/zach/good_svd_k.png')
