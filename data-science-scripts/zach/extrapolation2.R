
stop()
rm(list=ls(all=T))
gc(reset=T)
##########################################
# Libraries
##########################################
library(data.table)
library(datarobot)
library(pbapply)
library(ggplot2)
library(forecast)
library(readr)

colors <- c(
  "#1f78b4", "#ff7f00", "#6a3d9a", "#33a02c", "#e31a1c", "#b15928",
  "#a6cee3", "#fdbf6f", "#cab2d6", "#b2df8a", "#fb9a99", "#ffff99", "black",
  rep("grey", 100)
)

##########################################
# Data
##########################################

#Split
train <- fread('https://s3.amazonaws.com/datarobot_public_datasets/time_series/weekly_earth_co2_train.csv')
pred <- fread('https://s3.amazonaws.com/datarobot_public_datasets/time_series/weekly_earth_co2_test.csv')
train[,date := as.Date(date)]
pred[,date := as.Date(date)]
trainset_max <- train[,max(y)]
ggplot(train, aes(x=date, y=y)) + geom_line() + theme_bw()

##########################################
# ts models
##########################################

train_ts <- ts(train[['y']], frequency = 52, start=c(1959, 0))
test_ts <- ts(pred[['y']], frequency = 52, start=end(train_ts) + c(0,1))

model_arima <- auto.arima(train_ts, ic='bic', stepwise=FALSE)
f <- forecast(model_arima, length(test_ts))
plot(f)
lines(test_ts)
accuracy(f, test_ts)

# model_ets <- ets(train_ts, ic='bic')
# f <- forecast(model_ets, 12*15)
# plot(f)
# lines(test_ts)
# accuracy(f, test_ts)

##########################################
# OTP
##########################################

#Start project
metric <- 'RMSE'
projectObject <- SetupProject(train, projectName='CO2-weekly-OTP', maxWait=3600)
partition <- CreateDatetimePartitionSpecification(
  datetimePartitionColumn = 'date'
)
st <- SetTarget(
  project = projectObject,
  target = "y",
  metric=metric,
  partition=partition)
up <- UpdateProject(projectObject, workerCount = 20, holdoutUnlocked = TRUE)
ViewWebProject(projectObject)

#Run every autopilot model at 100%
Sys.sleep(60*5)
models <- GetAllModels(projectObject)
new <- lapply(models, function(mod){
  bp = list(
    blueprintId = mod,
    projectId = projectObject$projectId
  )
  tryCatch({
    RequestFrozenDatetimeModel(
      mod,
      trainingStartDate=min(train[['date']]),
      trainingEndDate=max(train[['date']]),
      )
  }, error=function(e) warning(e))
})

#Predict
Sys.sleep(60*5)
pd = UploadPredictionDataset(projectObject, pred, maxWait=3600)
models <- GetAllModels(projectObject)
frozen <- sapply(models, '[[', 'isFrozen')
models <- models[frozen]
pred_jobs_100pct <- pblapply(models, function(i){
  Sys.sleep(5)
  RequestPredictionsForDataset(projectObject, i$modelId, pd$id)
})
pred_list <- pblapply(seq_along(pred_jobs_100pct), function(i){
  messages <- suppressMessages({
    out <- data.table(
      pred,
      trainset_max = trainset_max,
      id = i,
      model = models[i][[1]][['modelType']],
      pred = GetPredictions(projectObject, pred_jobs_100pct[i], maxWait=3600)
    )
  })
  return(out)
})

#Plot the predictions
plot_dat <- rbindlist(pred_list)
plot_dat <- melt.data.table(
  plot_dat,
  id.vars=c('date', 'id', 'model'),
  measure.vars=c('y', 'trainset_max', 'pred')
)
plot_dat[model == 'R Based ARIMA' & variable =='pred', value := f$mean]
plot_dat <- plot_dat[!model %in% c('TensorFlow Neural Network Regressor'),]
keep = c('R Based ARIMA', 'Nystroem Kernel SVM Regressor - Forest (5x)')
#plotdat <- plotdat[model %in% keep,]
plot_dat[,model := paste(model, id)]
#plot_dat[,sort(unique(model))]
ggplot(plot_dat, aes(x=date, y=value, col=variable)) +
  geom_line() + facet_wrap(~model) + theme_bw() +
  scale_colour_manual(values=colors) +
  theme(legend.position = "top", strip.text = element_text(size=8)) +
  ggtitle('OTP Results')
