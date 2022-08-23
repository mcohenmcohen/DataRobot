library(Matrix)
set.seed(42)
nrow <- 1e4
ncol <- 1e3
nfactor <- 10
X <- rsparsematrix(nrow, ncol, density=0.1)
factors <- matrix(runif(ncol*nfactor), ncol=nfactor)