library(datarobot)
library(data.table)
library(readr)
library(C50)  # For example dataset
data(churn)

# Prepare data
dr_train <- churnTrain
dr_test <-  churnTest

trainfile <- 'train.csv'
testfile  <- 'test.csv'

write_csv(dr_train, trainfile)
write_csv(dr_test , testfile)

# Connect
ConnectToDataRobot(endpoint = 'https://app.datarobot.com/api/v2', token = 'PUT_YOUR_TOKEN_HERE')

# Start a new project
projectObject <- SetupProject(dataSource=trainfile)
up <- UpdateProject(projectObject, workerCount = 20, holdoutUnlocked = TRUE)

# Start autopilot
METRIC <- 'LogLoss'
st <- SetTarget(
  project = projectObject, target = "churn", metric=METRIC,
  accuracyOptimizedBlueprints=TRUE,
  maxWait=600)

# Upload prediction dataset
pred_data = UploadPredictionDataset(projectObject, testfile, maxWait=3600)

# Run every autopilot model at max AP
models <- ListModels(projectObject)
re_run <- unique(sapply(models, '[[', 'blueprintId'))
new <- lapply(re_run, function(mod){
  bp = list(
    blueprintId = mod,
    projectId = projectObject$projectId
  )
  tryCatch({
    RequestNewModel(projectObject$projectId, bp, scoringType='crossValidation')
  }, error=function(e) warning(e))
})

# Run every repository model at max AP
bps <- ListBlueprints(projectObject)
new <- lapply(bps, function(bp){
  tryCatch({
    RequestNewModel(projectObject$projectId, bp, scoringType='crossValidation')
  }, error=function(e) warning(e))
})

# Monitor progress
ViewWebProject(projectObject)

# Wait 
WaitForAutopilot(projectObject)

# Blend at max autopilot sample% - best by LogLoss
models <- ListModels(projectObject)
samplePct <- sapply(models, '[[', 'samplePct')
model_cats <- sapply(models, '[[', 'modelCategory')
stage_1 <- min(samplePct)
stage_3 <- max(samplePct[samplePct<100])
stage_2 <- setdiff(unique(samplePct), c(stage_1, stage_3, 100))
to_blend <- models[samplePct == stage_3 & model_cats == 'model']
logloss <- sapply(to_blend, function(x) x$metrics[[METRIC]]$crossValidation)
ids <- sapply(to_blend, '[[', 'modelId')
ids <- ids[order(logloss)]
number_of_models_to_blend <- sort(unique(c(2:25, 40, 50, length(ids))))
number_of_models_to_blend <- number_of_models_to_blend[number_of_models_to_blend <= length(ids)]
for(n in number_of_models_to_blend){
  for(b in datarobot::BlendMethods){
    tryCatch({
      RequestBlender(projectObject, ids[1:n], b)
    }, error=function(e) warning(e))
  }
}

# Todo: blend random sets of models

# Todo: download validation predictions, and blend models based on logloss penalized by correlation with best model

# Wait 
WaitForAutopilot(projectObject)

# Select best model
models <- ListModels(projectObject)
bp <- sapply(models, '[[', 'blueprintId')
samplePct <- sapply(models, '[[', 'samplePct')
logloss <- sapply(models, function(x) x$metrics[[METRIC]]$crossValidation)
models <- models[order(logloss)]
best_model_at_max_autopilot <- models[samplePct < 100][[1]]

# Feature impact + run at 100%
RequestFeatureImpact(best_model_at_max_autopilot)
RequestSampleSizeUpdate(best_model_at_max_autopilot, 100)

# Wait 
WaitForAutopilot(projectObject)

# Retrieve 100% model
models <- ListModels(projectObject)
bp <- sapply(models, '[[', 'blueprintId')
samplePct <- sapply(models, '[[', 'samplePct')
best_model_at_100 <- models[samplePct == 100 & bp == best_model_at_max_autopilot$blueprintId]
stopifnot(length(best_model_at_100) == 1)
best_model_at_100 <- best_model_at_100[[1]]

# View best model and calcualte feature impact
ViewWebModel(best_model_at_100)
RequestFeatureImpact(best_model_at_100)

# Predict on test set
pred_job <- RequestPredictionsForDataset(projectObject, best_model_at_100[['modelId']], pred_data$id)

# Download predictions
preds <- GetPredictions(projectObject, pred_job, maxWait=3600, type='probability')
dr_test_with_preds <- data.table(
  dr_test,
  pred = preds
  )

# Test accuract
library(Metrics)
eps <- 1e-6
dr_test_with_preds[,logLoss(churn=='yes', pmin(pmax(pred, 0+eps), 1-eps))]
dr_test_with_preds[,auc(churn=='yes', pred)]
