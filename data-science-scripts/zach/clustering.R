library(cluster)
library(fpc)
library(data.table)
library(ggplot2)
library(scales)

#Load data
red <- read.table('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep = ';', header = TRUE)
white <- read.table('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', sep = ';', header = TRUE)
red$red <- 1
white$red <- 0
wine <- rbind(red, white)

#Split X/Y
exclude <- 'red'
xnames <- setdiff(names(wine), exclude)

X <- wine[, xnames]

#Center and scale
X_scale <- scale(X, center = TRUE, scale = TRUE)

#PCA
X_PCA <- prcomp(X_scale)$x

#Find red/white best sep
dc <- discrcoord(X_PCA, wine$red, pool='n')$proj
mvc <- mvdcoord(X_PCA, wine$red, pool='n')$proj

#Kmeans
assemble_output <- function(x, name=''){
  data.frame(
    dc1 = dc[,1],
    dc2 = dc[,2],
    pca1 = X_PCA[,1],
    pca2 = X_PCA[,2],
    mvc1 = mvc[,1],
    mvc2 = mvc[,2],
    cluster = kmeansruns(x, krange=2:10, critout=TRUE, criterion='ch')$cluster,
    name=name
  )
}
dat <- list(
  'truth' = data.frame(
    dc1 = dc[,1],
    dc2 = dc[,2],
    pca1 = X_PCA[,1],
    pca2 = X_PCA[,2],
    mvc1 = mvc[,1],
    mvc2 = mvc[,2],
    cluster = wine$red,
    name='truth'
  ),
  'kmeans_raw' = assemble_output(X, 'raw'),
  'kmeans_scale' = assemble_output(X_scale, 'scaled'),
  'kmeans_pca2' = assemble_output(X_PCA[,1:2], 'pca2')
)

dat <- rbindlist(dat)
dat$cluster <- factor(dat$cluster)

#Plot
ggplot(dat, aes(x=dc1, y=dc2, color=cluster)) +
  geom_point(alpha=0.75) +
  facet_wrap(~name, scales='free') +
  theme_bw() + scale_colour_manual(
    values=c(
      '#1f78b4',
      '#ff7f00',
      '#6a3d9a',
      '#33a02c',
      '#e31a1c',
      '#b15928',
      '#ffff99'
    )
  )
