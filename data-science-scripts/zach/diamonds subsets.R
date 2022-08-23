library(ggplot2)
for(i in 2:4){
  n <- 10^i
  for(m in c(n, 5*n)){
    file <- paste0('~/datasets/diamonds', m, '.csv.gz')
    print(file)
    write.csv(head(diamonds, m), gzfile(file))
  }
}

for(n in 2:4){
  m <- n * 1000
  file <- paste0('~/datasets/diamonds', m, '.csv.gz')
  print(file)
  write.csv(head(diamonds, m), gzfile(file))
}

m <- 2500
file <- paste0('~/datasets/diamonds', m, '.csv.gz')
print(file)
write.csv(head(diamonds, m), gzfile(file))
