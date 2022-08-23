
stop()
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

#Make dataset
dat <- fread('~/datasets/births-per-10000-of-23-year-old-.csv')
setnames(dat, c('Year', 'Births'))
dat[,Date := as.Date(paste(Year, '01', '01', sep='-'))]
dat[,set := 'pred']
dat[Year < 1947, set := 'valid']
dat[Year < 1957, set := 'train']
dat[,table(year(date), set)]

#Add trend and month
dat[,year := year(date)]
dat[,month := paste0('m', month(date))]

#Add a weight
set.seed(42)
dat[,weight := 1000 + rnorm(.N)]

#Split
train <- dat[set != 'pred',]
pred <- dat[set == 'pred',]
trainset_max <- train[,max(co2)]
ggplot(train, aes(x=date, y=co2)) + geom_line() + theme_bw()
#write.csv(train, '~/datasets/test.csv', row.names = FALSE)

write_csv(train, '~/datasets/co2_train.csv')
write_csv(pred, '~/datasets/co2_test.csv')

##########################################
# ts models
##########################################

train_ts <- ts(train[['co2']], frequency = 12, start=c(1959, 1))
test_ts <- ts(pred[['co2']], frequency = 12, end=c(1997, 12))

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
metric <- paste('Weighted', metric)
projectObject <- SetupProject(train, projectName='CO2-OTP', maxWait=3600)
f1 <- CreateFeaturelist(projectObject, 'f1', c('year', 'month'))
partition <- CreateDatetimePartitionSpecification(
  datetimePartitionColumn = 'date'
)
st <- SetTarget(
  project = projectObject, target = "co2", metric=metric, weights='weight',
  partition=partition, featurelistId=f1$featurelistId)
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
pred <- rbindlist(pred_list)
pred <- melt.data.table(
  pred,
  id.vars=c('date', 'id', 'model'),
  measure.vars=c('co2', 'trainset_max', 'pred')
)
pred[model == 'TensorFlow Multilayer Perceptron Regressor', model := 'R Based ARIMA']
pred[model == 'R Based ARIMA' & variable =='pred', value := f$mean]
pred[,model := paste(model, id)]
pred[,table(model)]
ggplot(pred, aes(x=date, y=value, col=variable)) +
  geom_line() + facet_wrap(~model) + theme_bw() +
  scale_colour_manual(values=colors) +
  theme(legend.position = "top", strip.text = element_text(size=8)) +
  ggtitle('OTP Results')

##########################################
# TVH
##########################################

#Start project
metric <- 'RMSE'
projectObject <- SetupProject(train, projectName='CO2-TVH', maxWait=3600)
f1 <- CreateFeaturelist(projectObject, 'f1', c('trend', 'month'))
partition <- CreateUserPartition(
  userPartitionCol = 'set',
  validationType = 'TVH',
  trainingLevel = 'train',
  validationLevel = 'valid'
)
st <- SetTarget(
  project = projectObject, target = "co2", metric=metric,
  partition=partition, featurelistId=f1$featurelistId)
up <- UpdateProject(projectObject, workerCount = 20, holdoutUnlocked = TRUE)
ViewWebProject(projectObject)

#Run every autopilot model at 100%
Sys.sleep(60*10)
models <- GetAllModels(projectObject)
re_run <- unique(sapply(models, '[[', 'blueprintId'))
new <- lapply(re_run, function(mod){
  bp = list(
    blueprintId = mod,
    projectId = projectObject$projectId
  )
  tryCatch({
    RequestNewModel(projectObject$projectId, bp, samplePct=100)
  }, error=function(e) warning(e))
})

#Predict
Sys.sleep(60*10)
pd = UploadPredictionDataset(projectObject, pred, maxWait=3600)
models <- GetAllModels(projectObject)
sample_sizes <- sapply(models, function(x) x$samplePct)
models <- models[sample_sizes == 100]
pred_jobs_100pct <- pblapply(models, function(i){
  RequestPredictionsForDataset(projectObject$projectId, i$modelId, pd$id)
})
pred_list <- pblapply(seq_along(pred_jobs_100pct), function(i){
  out <- data.table(
    pred,
    trainset_max = trainset_max,
    id = i,
    model = models[i][[1]][['modelType']],
    pred = GetPredictions(projectObject, pred_jobs_100pct[i], maxWait=3600)
  )
})

#Plot the predictions
pred <- rbindlist(pred_list)
pred[,list(RMSE=sqrt(mean((pred - as.numeric(co2))^2))), by='model'][order(RMSE),]
pred <- melt.data.table(
  pred,
  id.vars=c('date', 'id', 'model'),
  measure.vars=c('co2', 'trainset_max', 'pred')
)
pred[,model := paste(model, id)]
pred <- pred[model != 'TensorFlow Multilayer Perceptron Regressor 11',]
ggplot(pred, aes(x=date, y=value, col=variable)) +
  geom_line() + facet_wrap(~model) + theme_bw() +
  scale_colour_manual(values=colors) +
  theme(legend.position = "bottom", strip.text = element_text(size=8)) +
  ggtitle('TVH Results')


