library(data.table)
data(iris)
iris = data.table(iris)
setnames(iris, 'Petal.Length', '花弁の長さ')
fwrite(iris, '~/datasets/train.csv')

iris[, 花弁の長さ := rep('ひも', .N)]
fwrite(head(iris), '~/datasets/test.csv')
