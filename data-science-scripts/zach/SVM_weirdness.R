library(data.table)
pred = fread('~/workspace/data-science-scripts/zach/SVMTest_Nystroem_Kernel_SVM_Classifier_(51)_64_Informative_Features_test.csv.csv')
act = fread('~/workspace/data-science-scripts/zach/test-1.csv')

trainset_mean = 0.15628
pred[,mean(Prediction)]
pred[,summary(Prediction)]


train = fread('~/Downloads/early_2012_2013_train.csv')
