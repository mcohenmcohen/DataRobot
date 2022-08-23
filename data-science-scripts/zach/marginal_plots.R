
#Setup data
set.seed(1)
library(ggplot2)
library(randomForest)
data(diamonds)
N <- nrow(diamonds)
all_rows <- 1:N
train_rows <- sample(all_rows, N*.66)
test_rows <- setdiff(all_rows, train_rows)

#Add fake year
diamonds$year <- sample(2000:2011, N, replace=TRUE)

#Split X/Y
tagret <- 'price'
xvars <- c('carat', 'clarity', 'color', 'cut', 'depth', 'table')
train_X <- diamonds[train_rows,xvars]
train_Y <- diamonds[train_rows,tagret]

#Fit model
model <- randomForest(train_X, train_Y, ntree=100, mtry=2)

#Find important vars
imp <- importance(model)
imp <- imp[order(imp[,1], decreasing=TRUE),]
imp_vars <- names(imp)
imp_vars <- setdiff(imp_vars, 'year')
imp_values <- round(imp/sum(imp) * 100, 1)

#Predict for test set
test_set  <- diamonds[test_rows, ]
test_set$pred <- predict(model, test_set[,xvars])

#Split numeric / cat
nums <- sapply(diamonds, is.numeric)
nums <- names(nums)[nums]
cats <- sapply(diamonds, is.factor) | sapply(diamonds, is.character)
cats <- names(cats)[cats]

nums <- intersect(nums, imp_vars)
cats <- intersect(cats, imp_vars)

#Data robot does all of the above, so start here for real
#(Will need to manually copy important vars from datarobot)
pdf('~/Desktop/plots_numeric.pdf', width=11, height=8.5)
for(var in nums){
  imp <- imp_values[var]
  plot_title <- paste0(
    '"', var,
    '" predicted vs actual by year (',
    imp, '% importance)')
  my_plot <- ggplot(test_set, aes_string(x = var)) +
    geom_point(aes_string(y=tagret), col='black', alpha=.05) +
    geom_smooth(aes_string(y=tagret), col='black') +
    geom_smooth(aes_string(y='pred'), col='red') +
    facet_wrap(~year) +
    ggtitle(plot_title) +
    scale_y_log10() +
    theme_bw()
  print(my_plot)
}
dev.off()
system('open ~/Desktop/plots_numeric.pdf')

pdf('~/Desktop/plots_cats.pdf', width=11, height=8.5)
for(var in cats){
  imp <- imp_values[var]
  plot_title <- paste0(
    '"', var,
    '" predicted vs actual by year (',
    imp, '% importance)')
  my_plot <- ggplot(test_set, aes_string(x = var)) +
    geom_boxplot(aes_string(y=tagret), col='black', alpha=.05) +
    geom_boxplot(aes_string(y='pred'), col='red', alpha=.05) +
    facet_wrap(~year) +
    ggtitle(plot_title) +
    scale_y_log10() +
    theme_bw()
  print(my_plot)
}
dev.off()
system('open ~/Desktop/plots_cats.pdf')
