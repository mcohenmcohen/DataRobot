library(data.table)
library(caTools)
dat <- fread('~/Downloads/train.csv_eXtreme_Gradient_Boosted_Trees_Classifier_with_Ear_(15)_100_Usable.csv')

dat[,colAUC(`Cross-Validation Prediction`, target), by='Partition']
dat[,colAUC(`Cross-Validation Prediction`, target)]
