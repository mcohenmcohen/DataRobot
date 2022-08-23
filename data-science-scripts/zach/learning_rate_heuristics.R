library(data.table)
data <- fread('https://s3.amazonaws.com/datarobot_public_datasets/lr_vs_loss.csv')
smoother <- loess(loss ~ log(learning_rate), data, span=.5)
data[,smooth := predict(smoother)]
plot(loss ~ log(learning_rate), data)
lines(smooth ~ log(learning_rate), data)

data[,diff_smooth := c(NA, diff(smooth))]

print(data[which.min(diff_smooth),])
points(smooth ~ log(learning_rate), data[which.min(diff_smooth),], col='red', pch=16)

plot(loss ~ learning_rate, data)
lines(smooth ~ learning_rate, data)
points(smooth ~ learning_rate, data[which.min(diff_smooth),], col='red', pch=16)
