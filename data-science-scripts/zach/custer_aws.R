library(data.table)
library(reshape2)
library(text2vec)
library(fpc)
library(ggplot2)
library(ggthemes)
library(Rtsne)
dat <- fread('~/workspace/data-science-scripts/zach/aws_dataset_sorted.csv')

# Cleanup
dat[,Service := gsub('.amazonaws.com', '', Service, fixed=T)]
dat[,ServiceEvent := paste0(Service, ':', Event)]

# Rehsape wide
dat_wide <- dcast.data.table(dat, User ~ ServiceEvent, value.var = 'Count')

# NAs here are zeros
dat_wide[is.na(dat_wide)] <- 0

# Make a matrix
stats <- as.matrix(dat_wide[,2:ncol(dat_wide), with=F])
stats <- stats[,colSums(sign(stats)) > 3]

# L2 normalize each row
stats <- text2vec::normalize(stats, norm='l1')

# PCA components
stats_pca <- prcomp(stats, retx = T)$x

# TSNE
stats_tsne <- Rtsne(stats_pca[,1:3], pca = F, verbose = T, check_duplicates = F, Rtsne=0, perplexity=40)$Y
colnames(stats_tsne) <- c('TSNE1', 'TSNE2')

# Cluster
clusters <- kmeansruns(stats_tsne, krange=3, critout = T)

# Plot
plotdat <- data.table(
  User=dat_wide[,User],
  cluster=factor(clusters$cluster),
  stats_tsne
)
ggplot(plotdat, aes(x=TSNE1, y=TSNE2, color=cluster)) + geom_point() + theme_tufte()
