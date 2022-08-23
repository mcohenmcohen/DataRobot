library(gtools)
x = c('H', 'M', 'L', 'P')
for(i in seq_along(x)){
  out <- combinations(length(x), i, v=x)
  out <- apply(out, 1, paste, collapse='')
  print(out)
}