library(data.table)
library(matrixStats)
library(Matrix)
library(ggplot2)
library(scales)
library(viridis)
library(ggthemes)
# https://rud.is/b/2016/02/14/making-faceted-heatmaps-with-ggplot2/

bad <- fread('~/datasets/mnist_Decision_Tree_Classifier_(Gini)_(10)_16_Informative_Features_validation_and_holdout.csv')
# good <- fread('~/datasets/mnist_eXtreme_Gradient_Boosted_Trees_Classifier_with_Ear_(13)_16_Informative_Features_validation_and_holdout.csv')
# good <- fread('~/datasets/mnist_TensorFlow_Deep_Learning_Classifier_(6)_64_Informative_Features_validation_and_holdout.csv')
good <- fread('~/datasets/mnist_eXtreme_Gradient_Boosted_Trees_Classifier_with_Ear_(13)_64_Informative_Features_validation_and_holdout.csv')

stopifnot(nrow(bad) == nrow(good))

setnames(bad, make.names(names(bad)))
setnames(good, make.names(names(good)))

bad <- bad[Partition == '0.0',]
good <- good[Partition == '0.0',]

stopifnot(all(good[['target']] == bad[['target']]))
act <- good[['target']]

pred_cols <- paste0('Cross.Validation.Prediction.', 0:9)
bad <- as.matrix(bad[,pred_cols,with=F])
good <- as.matrix(good[,pred_cols,with=F])
bad <- apply(bad, 1, which.max) - 1
good <- apply(good, 1, which.max) - 1

sum(act == bad) / length(bad)
sum(act == good) / length(good)

cm_bad <- as.matrix(table(act, bad))
cm_good <- as.matrix(table(act, good))

# image(cm_bad)
# image(cm_good)

# image(Matrix(cm_bad), useAbs=F)
# image(Matrix(cm_good), useAbs=F)

png('~/Desktop/plot1.png', width=600, height=600)
ggplot(melt(cm_bad), aes(factor(bad, levels=0:9), factor(act, levels=9:0))) +
  geom_tile(aes(fill = log1p(value), color = log1p(value))) +
  geom_text(aes(label = value), color="white", size=rel(4)) +
  scale_x_discrete(expand = c(0, 0), position = "top") +
  scale_y_discrete(expand = c(0, 0)) +
  xlab('Predicted') + ylab('Actual') +
  scale_fill_viridis(option="viridis") +
  scale_color_viridis(option="viridis") +
  coord_equal() +
  theme_tufte(base_family="Helvetica") +
  theme(legend.position="none")
dev.off()

png('~/Desktop/plot2.png', width=600, height=600)
ggplot(melt(cm_good), aes(factor(good, levels=0:9), factor(act, levels=9:0))) +
  geom_tile(aes(fill = log1p(value), color = log1p(value))) +
  geom_text(aes(label = value), color="white", size=rel(4)) +
  scale_x_discrete(expand = c(0, 0), position = "top") +
  scale_y_discrete(expand = c(0, 0)) +
  theme(legend.position="none") +
  xlab('Predicted') + ylab('Actual') +
  scale_fill_viridis(name="# Events", label=comma) +
  coord_equal() +
  theme_tufte(base_family="Helvetica") +
  theme(legend.position="none")
dev.off()
