set.seed(42)
a = c(
  1, 2, "I AM AN ADDITIONAL NON NUMERIC VALUE", '-NaN', '-nan',
  'null', 'na', 'n/a', '#N/A', 'N/A', '?', '.',
  'Inf', 'INF', 'inf', '-inf', '-Inf', '-INF', ' ', 'None',
  '', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN',
  '1.#IND', '1.#QNAN', 'N/A', 'NA', 'NULL', 'NaN', 'nan'
)
b = sort(unique(a))
x = data.table(iris)
x[,test_data := a]
write.csv(x, '~/datasets/iris_5_missing.csv', row.names=FALSE, quote=FALSE)
system('cat ~/datasets/iris_5_missing.csv')

write.csv(x, '~/datasets/iris_6_missing.csv', row.names=FALSE, quote=TRUE)
system('cat ~/datasets/iris_6_missing.csv')

set.seed(42)
a = c(
  1, 2, "I AM AN ADDITIONAL NON NUMERIC VALUE",
  '-NaN', '-nan'
)
b = sort(unique(a))
x = data.table(iris)
x[,test_data := a]
write.csv(x, '~/datasets/iris_7_missing.csv', row.names=FALSE, quote=FALSE)
system('cat ~/datasets/iris_7_missing.csv')
