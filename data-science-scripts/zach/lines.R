find_line <- function(a, b){
  diff <- a - b
  m <- diff[2] / diff[1]
  b <- a[2] - m * a[1]
  return(c(m=m, b=b))
}

l1 <- find_line(
  c(1e6, 0.05),
  c(7e6, 0.15)
)

l2 <- find_line(
  c(1e6, 0.05),
  c(7e6, 0.30)
)

l1
l2

l1[['m']] * 1e6 + l1[['b']]
l1[['m']] * 1e7 + l1[['b']]
l1[['m']] * 3.5e7 + l1[['b']]
l1[['m']] * 1e8 + l1[['b']]
l1[['m']] * 1e9 + l1[['b']]

[1] 5.740161
[1] 5.667609
