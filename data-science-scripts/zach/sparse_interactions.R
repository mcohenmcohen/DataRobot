rm(list=ls(all=T))
gc(reset=T)
set.seed(42)
library(Matrix)

N <- 5000

acol <- 100
a <- Matrix(matrix(rpois(N*acol, 0.1), nrow=N))

bcol <- 1000
b <- Matrix(matrix(rpois(N*bcol, 0.1), nrow=N))

c <- Matrix(0, nrow=N, ncol=acol*bcol, byrow=T)
#c <- as(c, 'dgRMatrix')


# SLOOOOOW
for(i in 1:acol){
  for(j in 1:bcol){
    c[,(i-1)*bcol+j] <- a[,i,drop=F] * b[,j,drop=F]
  }
}

