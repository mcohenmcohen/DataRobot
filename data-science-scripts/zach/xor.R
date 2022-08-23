library(readr)
set.seed(42)
N <- 1000

a = sample(seq(0, 1, length=N))
b = sample(seq(0, 1, length=N))

y = xor(sign(a > .5), sign(b > .5))
table(y)
table(sign(a > .5), sign(b > .5), y)

out <- data.frame(y, a, b)

write_csv(out, '~/workspace/data-science-scripts/zach/xor.csv')