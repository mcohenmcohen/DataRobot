library(data.table)
train = fread('~/datasets/Train_SU63ISt.csv')
test = fread('~/datasets/Test_0qrQsBZ.csv')
train[,Datetime := strftime(strptime(Datetime, '%d-%m-%Y %H:%M', tz = "UTC"), tz = "UTC")]
test[,Datetime := strftime(strptime(Datetime, '%d-%m-%Y %H:%M', tz = "UTC"), tz = "UTC")]
train[is.na(Datetime),]
test[is.na(Datetime),]
sapply(train, anyNA)
sapply(test, anyNA)
fwrite(train, '~/datasets/ts_train.csv')
fwrite(test, '~/datasets/ts_test.csv')
