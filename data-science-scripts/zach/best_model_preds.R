library(data.table)
dat_raw = fread('http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests=598c7bb17423161098cfbb50&max_sample_size_only=false')
dat <- copy(dat_raw)
dat[,length(unique(Filename))]

# ID models with holdout scores
# Models trained into the holdout will be NA
dat[,best_holdout_Score := max(`Gini Norm_H`, na.rm=T), by=c('Filename', 'Blueprint')]

# Shrink retrains the top 3 models at 95% and makes predictions
# These models will have predictions scores
dat <- dat[!is.na(`Prediction Gini Norm`),]

# Choose model with best non-95% holdout score
dat[,blueprint_by_project := 1:.N, by='Filename']
dat <- dat[,best_model := which.max(best_holdout_Score), by='Filename']
dat <- dat[blueprint_by_project == best_model,]
dat[,length(unique(Filename))]

# Show out-of-sample predictions for these models
simple_results <- dat[,list(
  Filename, `Prediction Gini Norm`, `Prediction LogLoss`,
  `Prediction MAD`, `Prediction RMSE`, `Prediction RMSLE`)]
head(simple_results)
summary(simple_results)

