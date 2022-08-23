library(data.table)
data(iris)
iris <- data.table(iris)
setnames(iris, paste('Unnamed:', 1:ncol(iris)))
write.csv(iris, '~/datasets/iris_unnamed.csv', row.names=T)
write.table(iris, '~/datasets/iris_unnamed.tsv', quote=FALSE, sep='\t', col.names = NA, row.names=T)
write.csv(iris, '~/datasets/iris_unnamed_no_blank.csv', row.names=F)
write.table(iris, '~/datasets/iris_unnamed_no_blank.tsv', quote=FALSE, sep='\t', row.names=F)
system('head ~/datasets/iris_unnamed.csv')
system('head ~/datasets/iris_unnamed.tsv')
system('head ~/datasets/iris_unnamed_no_blank.csv')
system('head ~/datasets/iris_unnamed_no_blank.tsv')

library(data.table)
data(iris)
iris <- data.table(iris)
setnames(iris, 'Species', 'Unnamed: 0')
write.csv(iris, '~/datasets/iris_unnamed2.csv', row.names=T)
write.table(iris, '~/datasets/iris_unnamed2.tsv', quote=FALSE, col.names = NA, row.names=T)
write.csv(iris, '~/datasets/iris_unnamed2_no_blank.csv', row.names=F)
write.table(iris, '~/datasets/iris_unnamed2_no_blank.tsv', quote=FALSE, sep='\t', row.names=F)
system('head ~/datasets/iris_unnamed2.csv')
system('head ~/datasets/iris_unnamed2.tsv')
system('head ~/datasets/iris_unnamed2_no_blank.csv')
system('head ~/datasets/iris_unnamed2_no_blank.tsv')


https://s3.amazonaws.com/datarobot_public_datasets/iris_unnamed.csv
https://s3.amazonaws.com/datarobot_public_datasets/iris_unnamed.tsv
https://s3.amazonaws.com/datarobot_public_datasets/iris_unnamed_no_blank.csv
https://s3.amazonaws.com/datarobot_public_datasets/iris_unnamed_no_blank.tsv
https://s3.amazonaws.com/datarobot_public_datasets/iris_unnamed2.csv
https://s3.amazonaws.com/datarobot_public_datasets/iris_unnamed2.tsv
https://s3.amazonaws.com/datarobot_public_datasets/iris_unnamed2_no_blank.csv
https://s3.amazonaws.com/datarobot_public_datasets/iris_unnamed2_no_blank.tsv
