stop()
rm(list=ls(all=T))
gc(reset=T)
library(data.table)
library(compiler)
N <- 1.5e9  # 3e9
Wild <- cmpfun(function(x){
  10 * sin(0.3 * x) * sin(1.3 * x^2) +
    0.00001 * x^4 + 0.2 * x + 80
})
# plot(Wild, -50, 50, n=1000)

out <- data.table(
  x=seq(-50, 50, length=N)
)
gc(reset=T)

out[, y := 10 * sin(0.3 * x)]
gc(reset=T)

out[, y := y * sin(1.3 * x^2)]
gc(reset=T)

out[, y := y +  0.00001 * x^4]
gc(reset=T)

out[, y := y + 0.2 * x + 80]
gc(reset=T)

out[, x := round(x)]
gc(reset=T)

out[, y_out := round(y)]
gc(reset=T)
fname <- '~/wild_funtion.csv'
fwrite(out[,list(x, y=y_out)], fname)
utils:::format.object_size(file.size(fname), "auto")
gc(reset=T)

set.seed(1234)
out[, y_out := round(y + rnorm(.N, sd=10))]
fwrite(out[,list(x, y=y_out)], fname, append=T, col.names=FALSE)
utils:::format.object_size(file.size(fname), "auto")

set.seed(42)
out[, y_out := round(y + rnorm(.N, sd=10))]
fwrite(out[,list(x, y=y_out)], fname, append=T, col.names=FALSE)
utils:::format.object_size(file.size(fname), "auto")
