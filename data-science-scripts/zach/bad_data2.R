library(readr)
set.seed(42)
rows <- 1000
cols <- 10
x <- matrix(rnorm(rows*cols), ncol=cols)
cf <- rnorm(cols) * sample(0:1, cols, replace=TRUE)
y <- x %*% cf + rnorm(rows) * cols * .10
y <- y[,1]
y <- as.integer(y > median(y))
out <- data.table(y=y, x)
out[,V1 := ifelse(V1>median(V1), "TRUE", "FALSE")]
out[sample(1:.N, 3) ,V1 := ""]
out[,table(V1)]
write.csv(out, '~/datasets/TRUE_FALSE_BLANK.csv', quote=T, row.names=F)
