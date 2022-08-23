
##########################################
# Case 1
##########################################

set.seed(42)
nrows <- 1000
ncols <- 10

#Numeric
X <- matrix(runif(nrows * ncols), ncol=ncols)
Y <- X %*% rnorm(ncols)

mid <- floor(nrows/2)
X[mid:nrows,] <- NA
Y[1:(mid-1)] <- NA

dat <- data.frame(Y, X)
write.csv(dat, '~/datasets/bad_target.csv', row.names=FALSE)

#Should be ok
dat$A <- runif(nrows)
write.csv(dat, '~/datasets/weird_target.csv', row.names=FALSE)

#Text
ncols <- 100
X <- matrix(sample(letters, nrows * ncols, replace=TRUE), ncol=ncols)
Y <- as.integer(rowSums(X == 'z'))
X <- apply(X, 1, paste, collapse=' ')

X[mid:nrows] <- NA
Y[1:(mid-1)] <- NA

dat <- data.frame(Y, X)
write.csv(dat, '~/datasets/bad_target_text.csv', row.names=FALSE)

#Should be ok
dat$X2 <- runif(nrows)
write.csv(dat, '~/datasets/weird_target_text.csv', row.names=FALSE)

##########################################
# Case 2
##########################################

set.seed(42)
nrows <- 1000
ncols <- 10

#Numeric
X <- matrix(runif(nrows * ncols), ncol=ncols)
Y <- X %*% rnorm(ncols)

#Bad numeric
mid <- floor(nrows/2)
Bad <- runif(nrows)
Bad[(mid+1):nrows] <- NA
Y[1:(mid)] <- NA
dat <- data.frame(Y, X)
dat$Bad <- Bad
write.csv(dat, '~/datasets/bad_numeric.csv', row.names=FALSE)

#Bad text
words <- sapply(sample(3:8, 1000, replace=TRUE), function(x) paste(sample(letters, x, replace=TRUE), collapse=''))
Bad <- sapply(sample(1:20, nrows, replace=TRUE), function(x) paste(sample(words, x, replace=TRUE), collapse=' '))
mid <- floor(nrows/2)
Bad[(mid+1):nrows] <- NA
Y[1:(mid)] <- NA
dat <- data.frame(Y, X)
dat$Bad <- Bad
write.csv(dat, '~/datasets/bad_text.csv', row.names=FALSE)
