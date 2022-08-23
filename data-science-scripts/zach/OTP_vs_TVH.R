
########################################################
# Libraries
########################################################

library(data.table)
library(rmongodb)
library(jsonlite)
library(ggplot2)
library(shrinkR)

########################################################
# Load data
########################################################

TVH_raw <- shrinkR::getLeaderboard('58bc4a094e0d8527f38ae7ef')
OTP_raw <- shrinkR::getLeaderboard('58bc49352205b91d3848ce2e')

########################################################
# Sumamrize data
########################################################

TVH <- copy(TVH_raw)
setnames(TVH, make.names(names(TVH), unique = TRUE))
TVH[,sum(Prediction.dataset_name!='')]
TVH <- TVH[Prediction.dataset_name!='',]
TVH <- TVH[,list(
  Prediction.Gini.Norm.TVH = max(Prediction.Gini.Norm)
), by='Prediction.dataset_name']

OTP <- copy(OTP_raw)
setnames(OTP, make.names(names(OTP), unique = TRUE))
OTP[,sum(Prediction.dataset_name!='')]
OTP <- OTP[Prediction.dataset_name!='',]
OTP <- OTP[,list(
  Prediction.Gini.Norm.OTP = max(Prediction.Gini.Norm)
), by='Prediction.dataset_name']

########################################################
# Join
########################################################

out <- merge(TVH, OTP, by='Prediction.dataset_name')
out[,Prediction.dataset_name := gsub('https://s3.amazonaws.com/datarobot_public_datasets/', '', Prediction.dataset_name)]
plot(Prediction.Gini.Norm.OTP ~ Prediction.Gini.Norm.TVH, out)
abline(a=0, b=1)
out

########################################################
# Yung's Data
########################################################
y = fread('~/workspace/data-science-scripts/zach/wp36_tvh_datetime_tsfeatures.csv')
setnames(y, 'R2_Datetime', 'R2_OTP')
setnames(y, 'R2_Tvh', 'R2_TVH')
y[model_type == 'Nystroem Kernel SVM Regressor', model_type := 'Nystroem SVM']
y[model_type == 'eXtreme Gradient Boosted Trees Regressor with Early Stopping', model_type := 'XGboost']
y[model_type == 'RandomForest Regressor', model_type := 'RandomForest']
plot(R2_OTP ~ R2_TVH, y)
abline(a=0, b=1)
y

best <- y[,list(
  model_type = 'best_model',
  R2_OTP = max(R2_OTP, na.rm=T),
  R2_TVH = max(R2_TVH, na.rm=T)
), by='dataset']
plot(R2_OTP ~ R2_TVH, best)
abline(a=0, b=1)
best[,all(R2_OTP > R2_TVH)]
glogbest