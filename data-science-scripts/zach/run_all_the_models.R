#WARNING! This script will run hundreds of models (maybe like 300-400)
#Only run on small datasets!
library(datarobot)
gc(reset=TRUE)

#Connect to DataRobot
creds <- list(
  prod = list(
    endpoint = 'app.datarobot.com',
    token = 'YOUR_TOKEN_GOES_HERE'
  ),
  staging = list(
    endpoint = 'staging.datarobot.com',
    token = 'YOUR_TOKEN_GOES_HERE'
  ),
  ow = list(
    endpoint = '10.20.53.43',
    token = 'YOUR_TOKEN_GOES_HERE'
  )
)
connection_wrapper <- function(env){
  url <- paste0('https://', env[['endpoint']], '/api/v2')
  ConnectToDataRobot(endpoint = url, token = env[['token']])
}
connection_wrapper(creds[['prod']])

#Upload dataset
data(iris)
projectObject <- SetupProject(iris, projectName = 'iris', maxWait=3600)

#Start autopilot
st <- SetTarget(project=projectObject, target="Sepal.Length", metric='RMSE')
up <- UpdateProject(projectObject, workerCount = 20, holdoutUnlocked = TRUE)

#Run every repository model at max AP %
bps <- ListBlueprints(projectObject)
new <- lapply(bps, function(bp){
  tryCatch({
    RequestNewModel(projectObject$projectId, bp, scoringType='crossValidation')
  }, error=function(e) warning(e))
})

#Run every autopilot model at max AP %
models <- GetAllModels(projectObject)
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

#Monitor progress
ViewWebProject(projectObject)

# Wait
Sys.sleep(60*15)

#Blend best 2-50 models
models <- GetAllModels(projectObject)
samplePct <- sapply(models, '[[', 'samplePct')
model_cats <- sapply(models, '[[', 'modelCategory')
stage_1 <- min(samplePct)
stage_3 <- max(samplePct[samplePct<100])
stage_2 <- setdiff(unique(samplePct), c(stage_1, stage_3, 100))
blend <- models[samplePct == stage_3 & model_cats == 'model']
rmse <- sapply(blend, function(x) x$metrics[['RMSE']]$crossValidation)
ids <- sapply(blend, '[[', 'modelId')
ids <- ids[order(rmse)]
for(n in c(2:25, 40, 50)){
  for(b in BlendMethods){
    tryCatch({
      RequestBlender(projectObject, ids[1:n], b)
    }, error=function(e) warning(e))
  }
}

# Wait
Sys.sleep(60*15)

#Select best model
models <- GetAllModels(projectObject)
samplePct <- sapply(models, '[[', 'samplePct')
pred <- models[samplePct < 100]
rmse <- sapply(pred, function(x) x$metrics[['RMSE']]$crossValidation)
best <- pred[order(rmse)][[1]]

#Run best model at 100%
new_id <- RequestSampleSizeUpdate(best, 100)
best_at_100 <- GetModelFromJobId(projectObject, new_id)
