library(data.table)
library(caret)
d <- fread('~/Downloads/Diabetes_Readmitting_Prediction_Regularized_Logistic_Regression_(L2)_(6)_64_Valid_Features.csv')
setnames(d, make.names(names(d)))

d[,table(Partition)]

#d[,Cross.Validation.Prediction := round(Cross.Validation.Prediction, 4)]


uniq <- d[,sort(unique(Cross.Validation.Prediction))]
dec = d[,quantile(Cross.Validation.Prediction, 0:100/100), type=8]
dec

dec = c(0, dec, 100)
pred_bins = d[,cut(Cross.Validation.Prediction, dec, include.lowest = TRUE)]

d[,pred := Cross.Validation.Prediction > .80]
d[,pred := Cross.Validation.Prediction >= 0.792312]
d[,pred := Cross.Validation.Prediction >= 0.793]
d[,pred := Cross.Validation.Prediction > 0.3519]

d[Partition == 'Holdout', .N]
d[Partition == 'Holdout',confusionMatrix(
  pred,
  readmitted,
  positive='TRUE'
)]

d[Partition == '0.0', .N]
d[Partition == '0.0',confusionMatrix(
  pred,
  readmitted,
  positive='TRUE'
)]
