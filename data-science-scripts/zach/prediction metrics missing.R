library(data.table)
data = fread('http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests=5defd3174f0914000ba6154d&max_sample_size_only=false')
for (var in sort(names(data)[grepl('prediction', tolower(names(data)))])){
  if(all(is.na(data[[var]]))){
    print(var)
  }
}

sort(names(data)[grepl('Prediction', tolower(names(data)))])
summary(data[['Prediction LogLoss']])
summary(data[['Prediction Gini Norm']])
summary(data[['Prediction AUC']])
summary(data[['Prediction LogLoss']])
summary(data[['Prediction Tweedie Deviance']])

summary(data[['Gini Norm_H']])
summary(data[['AUC_H']])
summary(data[['LogLoss_H']])
summary(data[['Tweedie Deviance_H']])

