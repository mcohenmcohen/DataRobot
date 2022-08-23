set.seed(42)
library(data.table)
x = rgamma(10000, 3)
quantiles = 0:100/100
out <- data.table(
  input_data = quantile(x, quantiles),
  percentile = quantiles * 100,
  ridit = (quantiles - .50) * 2
)
out[, coefficient := ridit * 0.112967]
out[, value := input_data * coefficient]
fwrite(out, '~/data.csv')