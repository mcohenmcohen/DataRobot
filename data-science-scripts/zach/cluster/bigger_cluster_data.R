# Setup
rm(list=ls(all=T))
gc(reset=T)
library(MixSim)
library(data.table)
library(fpc)
N_clusters <- 10
N_ROWS = 3000000
N_COLS = 20

# Make simulated data
A <- MixSim(MaxOmega = 0, K = N_clusters, p = N_COLS)
sim_matrix <- simdataset(N_ROWS, A$Pi, A$Mu, A$S)
sim <- data.table(
  scale(sim_matrix$X, center=T, scale=T),
  true_cluster=paste0('c', sim_matrix$id)
)
sim[,list(.N), by='true_cluster']
# fwrite(data.table(sim_matrix$X), '~/Downloads/bigger_cluster_data.csv')

# SVD then kmeans
t1 <- Sys.time()
data_preprocessed <- scale(sim_matrix$X, center=T, scale=T)
data_preprocessed <- svd(data_preprocessed, nu=10, nv=10)$u
clusters_kmeans_svd <- kmeansruns(data_preprocessed, critout=T, criterion = 'ch', krange=2:N_clusters, iter.max=100, runs=10)
sim[,clusters_kmeans_svd := paste0('c', clusters_kmeans_svd$cluster)]
t2 <- Sys.time()
print(t2 - t1)
sim[,list(.N), by='clusters_kmeans_svd']
