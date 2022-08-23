set.seed(42)
x <- runif(100)
y <- x*2
x[1] <- NA
out <- data.table(y, x)
write.csv(out, '~/datasets/one_missing.csv', row.names=F)
out[['x']][1:5] <- NA
write.csv(out, '~/datasets/five_missing.csv', row.names=F)
out[['x']][1:9] <- NA
write.csv(out, '~/datasets/nine_missing.csv', row.names=F)
out[['x']][1:10] <- NA
write.csv(out, '~/datasets/ten_missing.csv', row.names=F)
out[['x']][1:11] <- NA
write.csv(out, '~/datasets/eleven_missing.csv', row.names=F)
