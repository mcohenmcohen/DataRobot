library(data.table)
library(readr)
data(iris)
iris <- data.table(iris)

write_csv(iris, '~/datasets/iris.csv')
bad <- copy(iris[Species=='versicolor',])
bad[,Species := "A NEW SPECIES OF IRIS"]

write_csv(iris[Species %in% c('versicolor', 'virginica'),], '~/datasets/iris_test_case_1.csv')
write_csv(iris[Species=='versicolor',], '~/datasets/iris_test_case_2.csv')
write_csv(bad, '~/datasets/iris_test_case_3.csv')

