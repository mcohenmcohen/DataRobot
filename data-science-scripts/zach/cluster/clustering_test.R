# Setup
library(data.table)
library(fpc)
library(ggplot2)
library(ggthemes)
set.seed(42)

# Info
# This script is to explore a bug we found with 2D geospatial clustering
# https://datarobot.atlassian.net/browse/MODEL-7708

# Make Data
centers <- list(
  c(-76, 45),
  c(32, 40),
  c(-76, -10),
  c(28, -26)
)

N <- 1000
data <- rbindlist(lapply(centers, function(center){
  data.table(
    x=rnorm(N, center[1], 5),
    y=rnorm(N, center[2], 5)
  )
}))
fwrite(data, '~/workspace/data-science-scripts/zach/cluster/example_geo_clusters_simple_xy.csv')

# Cluster
data_preprocessed <- data[,cbind(x, y)]
data_preprocessed <- scale(data_preprocessed, center=T, scale=T)
data_preprocessed <- svd(data_preprocessed, nu=2, nv=2)$u[,1:2]  # u[,1:2] = good | u[,1] = bad and replicates the DR issue
clusters <- kmeansruns(data_preprocessed, critout=T, criterion = 'asw', krange=4)
data[,cluster := factor(clusters$cluster)]

# Plot colors
colors <- c(
  "#1f78b4", "#ff7f00", "#6a3d9a", "#33a02c", "#e31a1c", "#b15928",
  "#a6cee3", "#fdbf6f", "#cab2d6", "#b2df8a", "#fb9a99", "black",
  "grey1", "grey10"
)

# Plot
ggplot(data, aes(x=x, y=y, color=cluster)) + geom_point() + scale_color_manual(values=colors) + theme_tufte()

# Compare to DR
# https://app.datarobot.com/projects/61d4af7259e96543d9d300f0/models/61d4bb09607f3961763f3ec8/make-predictions
dr_cluster <- fread('~/workspace/data-science-scripts/zach/cluster/example_geo_clusters_simple_xy.csv_K-Means_Clustering_(3)_100_Informative_Features.csv')
setnames(dr_cluster, gsub("Cross-Validation Prediction Cluster ", "C", names(dr_cluster)))
dr_cluster[,cluster := apply(cbind(C1, C2, C3, C4), 1, which.max)]
dr_cluster[,cluster := factor(cluster)]
ggplot(dr_cluster, aes(x=x, y=y, color=cluster)) + geom_point() + scale_color_manual(values=colors) + theme_tufte()
