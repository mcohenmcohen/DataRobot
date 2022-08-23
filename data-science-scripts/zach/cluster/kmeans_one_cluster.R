# Setup
rm(list=ls(all=T))
dev.off()
gc(reset=T)
library(MixSim)
library(data.table)
library(tsne)
library(fpc)
library(ggplot2)
library(ggthemes)
set.seed(42)
colors <- c(
  "#1f78b4", "#ff7f00", "#6a3d9a", "#33a02c", "#e31a1c", "#b15928",
  "#a6cee3", "#fdbf6f", "#cab2d6", "#b2df8a", "#fb9a99", "black",
  "grey1", "grey10"
)
N_clusters <- 5

# Reference
# https://library.ndsu.edu/ir/bitstream/handle/10365/26766/On%20K-Means%20Clustering%20Using%20Mahalanobis%20Distance.pdf?sequence=1&isAllowed=y

# Make simulated data
N <- 1000
x <- as.numeric(scale(rnorm(N, mean=0, sd=1), center=T, scale=T))
y <- as.numeric(scale(2*x + rnorm(N, mean=0, sd=1), center=T, scale=T))
sim <- data.table(x, y)
# ggplot(sim, aes(x=x, y=y)) + geom_point() + theme_tufte()

# kmeans on the raw data
clusters_kmeans_raw <- kmeansruns(sim[,cbind(x, y)], critout=T, criterion = 'ch', krange=N_clusters, iter.max=1000, runs=1000)
sim[,cluster_kmeans_raw := paste0('c', clusters_kmeans_raw$cluster)]
# ggplot(sim, aes(x=x, y=y, color=cluster_kmeans_raw)) +
#   geom_point() + scale_color_manual(values=colors) +
#   theme_tufte() + theme(legend.position='none') + ggtitle('Raw Data + Kmeans')

# kmeans on the SVD data
data_preprocessed <- sim[,cbind(x, y)]
data_preprocessed <- scale(data_preprocessed, center=T, scale=T)
data_preprocessed <- svd(data_preprocessed, nu=2, nv=2)$u[,1:2]
clusters_kmeans_svd <- kmeansruns(data_preprocessed, critout=T, criterion = 'ch', krange=N_clusters, iter.max=1000, runs=1000)
sim[,clusters_kmeans_svd := paste0('c', clusters_kmeans_svd$cluster)]
# plot_sim <- copy(sim)  # To plot on the svd axes
# plot_sim[,x := data_preprocessed[,1]]
# plot_sim[,y := data_preprocessed[,2]]
# ggplot(plot_sim, aes(x=x, y=y, color=clusters_kmeans_svd)) +
#   geom_point() + scale_color_manual(values=colors) +
#   theme_tufte() + theme(legend.position='none') + ggtitle('SVD Data + Kmeans')

# Plot
data_kmeans_raw <- data.table(sim[,list(x, y)], data='raw data + kmeans', cluster=paste0('c', clusters_kmeans_raw$cluster))
data_kmeans_svd <- data.table(sim[,list(x, y)], data='svd + kmeans', cluster=paste0('c', clusters_kmeans_svd$cluster))
plot_data <- rbind(data_kmeans_raw, data_kmeans_svd, fill=T, use.names=T)
ggplot(plot_data, aes(x=x, y=y, color=cluster)) +
  geom_point() + scale_color_manual(values=colors) +
  theme_tufte() + theme(legend.position="bottom") +
  facet_wrap(~data) + coord_cartesian()
