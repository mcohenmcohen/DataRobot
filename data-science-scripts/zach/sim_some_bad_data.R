
make_data <- function(n=10){

  cols=5

  library(data.table)
  x <- sapply(0:(cols-1), function(i){
    out <- rep(0L, n * cols)
    out[(i*n + 1):(i*n+n)] <- 1L
    return(out)
  })

  #Calculate y
  CF <- rnorm(cols)
  y <- x %*% CF + (rnorm(n) / 10) * cols
  dat <- data.table(y=y[,1], x)

  #NA / 1
  dat[V1 == 0, V1 := NA]

  #0 / NA
  dat[V2 == 1, V2 := NA]

  #0 / small
  dat[, V3 := as.numeric(V3)]
  dat[V3 == 0, V3 := 1e-9]

  #0 / large
  dat[, V4 := as.numeric(V3)]
  dat[V4 == 1, V4 := 1e+9]

  #small / large
  dat[, V5 := as.numeric(V5)]
  dat[V5 == 0, V5 := 1e-9]
  dat[V5 == 1, V5 := 1e+9]

  return(dat)
}

dat <- make_data(1000)
write.csv(
  dat, paste0('~/datasets/stress_test_', nrow(dat), '.csv'),
  row.names=FALSE
)
