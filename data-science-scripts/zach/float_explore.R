library(data.table)
data(iris)

dat = data.table(iris)
dat[Species == 'setosa', Species := '1.0']
fwrite(dat, '~/datasets/iris_float_in_species.csv')
dat = dat[Species == '1.0']
fwrite(dat, '~/datasets/iris_float_only.csv')

dat = data.table(iris)
dat[Species == 'setosa', Species := '1']
fwrite(dat, '~/datasets/iris_int_in_species.csv')
dat = dat[Species == '1']
fwrite(dat, '~/datasets/iris_int_only.csv')

dat = data.table(iris)
dat = dat[Species != 'setosa',]
fwrite(dat, '~/datasets/iris_no_setosa.csv')

dat = data.table(iris)
dat = dat[,Species := as.integer(Species)]
fwrite(dat, '~/datasets/iris_int.csv')