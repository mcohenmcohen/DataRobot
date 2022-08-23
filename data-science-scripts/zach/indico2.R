rm(list=ls(all=T))
gc(reset=T)
set.seed(42)

library(data.table)
library(datarobot)

train <- fread('https://s3.us-east-2.amazonaws.com/indicodrtest/AugmentedTrain.csv')
train[,c('V1', 'id', 'qid1', 'qid2') := NULL]
test <- fread('https://s3.us-east-2.amazonaws.com/indicodrtest/AugmentedTest.csv')
test[,c('V1', 'test_id') := NULL]
dat <- rbind(train, test, fill=T)
dat[,ID := 1:.N]
rm(train, test)
gc(reset=T)

TARGET_MEAN = 0.165
sum_zero <- dat[!is.na(is_duplicate),sum(is_duplicate == 0)]
sum_one  <- dat[!is.na(is_duplicate),sum(is_duplicate == 1)]
reweight <-  (- sum_zero * TARGET_MEAN) / (sum_one * TARGET_MEAN - sum_one)

actual_mean = dat[!is.na(is_duplicate),mean(is_duplicate)]
dat[,weight := 1]
dat[is_duplicate == 1, weight := reweight]
new_mean <- dat[!is.na(is_duplicate), sum(is_duplicate * weight) / sum(weight)]
stopifnot(all.equal(new_mean, TARGET_MEAN))

metric = 'Weighted LogLoss'

#Duplicates
all_questions = dat[,data.table(q=c(question1, question2), key='q')]
all_questions = all_questions[,list(q1n=.N), by='q']
setkeyv(all_questions, 'q1n')
setorderv(all_questions, 'q1n')
all_questions[,q1id := paste0('q', 1L:.N)]
setkeyv(all_questions, 'q')
dat <- merge(dat, all_questions, by.x='question1', by.y='q')
setnames(all_questions, c('q1id', 'q1n'), c('q2id', 'q2n'))
dat <- merge(dat, all_questions, by.x='question2', by.y='q')
dat[,max_n := pmax(q1n, q2n)]
dat[,min_n := pmin(q1n, q2n)]
setkeyv(dat, 'ID')
setorderv(dat, 'ID')

#Resplit train/test
dat[,c('ID') := NULL]
train <- dat[!is.na(is_duplicate),]
test <- dat[is.na(is_duplicate),]
test[,is_duplicate := NULL]
rm(dat, all_questions)
gc(reset=T)

##########################################
# DataRobot - LogLoss
##########################################

#Connection creds
creds <- list(
  prod = list(
    endpoint = 'app.datarobot.com',
    token = 'ZSa8G0D6TzCYUmVqDh4TB25dU0Cq8tyd'
  ),
  staging = list(
    endpoint = 'staging.datarobot.com',
    token = 'zSKi9K2Kr6_26r8sBRYniLx9bub1_LYq'
  ),
  ow = list(
    endpoint = '10.20.53.43',
    token = 'a2IIThlDl2XMvowC43ll1hagVPT6U3ZZ'
  ),
  pluni = list(
    endpoint = 'uniapp.datarobot.com',
    token = 'ZSa8G0D6TzCYUmVqDh4TB25dU0Cq8tyd'
  )
)

#Connect
connection_wrapper <- function(env){
  url <- paste0('https://', env[['endpoint']], '/api/v2')
  ConnectToDataRobot(endpoint = url, token = env[['token']])
}
connection_wrapper(creds[['prod']])
#load('output/project.rda'); load('output/best_model.rda')

#Setup project
partition <- CreateStratifiedPartition(
  validationType='CV',
  holdoutPct = 0,
  validationPct = 10,
  reps = 5
)
projectObject <- SetupProject(dataSource = train, projectName = 'indoco-magic-weight-dr', maxWait=3600)

#Load an old project
#projectObject <- datarobot::GetProject('58c88512c808914818b90b05')

#Start autopilot
st <- SetTarget(
  project = projectObject, target = "is_duplicate", weights = 'weight',
  metric=metric,
  partition=partition, maxWait=600)
up <- UpdateProject(projectObject, workerCount = 20, holdoutUnlocked = TRUE)

#Run every autopilot model at max AP
#Consider  samplePct=100.  Could overfit blenders
models <- GetAllModels(projectObject)
re_run <- unique(sapply(models, '[[', 'blueprintId'))
new <- lapply(re_run, function(mod){
  bp = list(
    blueprintId = mod,
    projectId = projectObject$projectId
  )
  tryCatch({
    RequestNewModel(projectObject$projectId, bp)
  }, error=function(e) warning(e))

})

#Run every repository model at max AP
bps <- ListBlueprints(projectObject)
new <- lapply(bps, function(bp){
  tryCatch({
    RequestNewModel(projectObject$projectId, bp)
  }, error=function(e) warning(e))
})

#Monitor progress
#Current best: 0.3580 @80%
ViewWebProject(projectObject)

# Blend some models
Sys.sleep(6*60*60)

#Blend at max autopilot sample% - best by LogLoss
models <- GetAllModels(projectObject)
samplePct <- sapply(models, '[[', 'samplePct')
fl <- sapply(models, '[[', 'featurelistId')
model_cats <- sapply(models, '[[', 'modelCategory')
stage_1 <- min(samplePct)
stage_3 <- max(samplePct[samplePct<100])
stage_2 <- setdiff(unique(samplePct), c(stage_1, stage_3, 100))
blend <- models[samplePct == stage_3 & model_cats == 'model']
error <- sapply(blend, function(x) x$metrics[[metric]]$validation)
ids <- sapply(blend, '[[', 'modelId')
ids <- ids[order(error)]
#blend_to_run <- BlendMethods
blend_to_run <- c('GLM')
for(n in c(2, 10, 20, min(length(ids), 40))){
  for(b in blend_to_run){
    tryCatch({
      RequestBlender(projectObject, ids[1:n], b)
    }, error=function(e) warning(e))
  }
}

# Re-run best model at 100%
#Select best model
Sys.sleep(4*60*60)
models <- GetAllModels(projectObject)
samplePct <- sapply(models, '[[', 'samplePct')
fl <- sapply(models, '[[', 'featurelistName')
pred <- models[samplePct < 100]
error <- sapply(pred, function(x) x$metrics[[metric]]$validation)
best <- pred[order(error)][[1]]

#Run a 100%
RequestFeatureImpact(best)
new_id <- RequestSampleSizeUpdate(best, 100)
best_at_100 <- GetModelFromJobId(projectObject, new_id)
save(new_id, best_at_100, file='output/best_model.rda')

# View best model
# best_at_100 <- GetModelObject(projectObject, '58f226d6c808916b46927a6d')
# ViewWebModel(best_at_100)
RequestFeatureImpact(best_at_100)
error <- cmpfun(function(a,b) sqrt(mean((a-b)^2)))

##########################################
# Chunk and upload pred data
##########################################

d <- test[is.na(is_duplicate),]
chunks <- split(d, sort(1:nrow(d) %% 100))
files <- pbsapply(seq_along(chunks), function(i){
  fn <- paste0('output/sub_chunk_', i, '.csv')
  write_csv(chunks[[i]], fn)
  return(fn)
})
pred_datasets <- pblapply(files, function(f){
  s <- gc(reset=T)
  UploadPredictionDataset(projectObject, f, maxWait=3600)
})
save(files, pred_datasets, file='output/pred_stuff.rda')

#Select best model
models <- GetAllModels(projectObject)
RMSE <- sapply(models, function(x) x$metrics[[metric]]$holdout)
models <- models[order(RMSE)]
ids <- sapply(models, function(x) x$blueprintId)
sample_sizes <- sapply(models, function(x) x$samplePct)
best_non_100 <- models[sample_sizes < 100][[1]]
best_non_100_at_100 <- models[sample_sizes == 100 & ids == best_non_100$blueprintId]

# best_non_100_at_100 <- list(GetModelObject(projectObject, '58f226d6c808916b46927a6d'))
# ViewWebModel(best_non_100_at_100)

pred_jobs_100pct <- pblapply(pred_datasets, function(i){
  if(length(best_non_100_at_100) > 0){
    model <- best_non_100_at_100[[1]]
  } else{
    warning('No 100% model yet')
    model <- best_non_100
  }
  pred_job <- RequestPredictionsForDataset(i$project, model$modelId, i$id)
  Sys.sleep(10)
  return(pred_job)
})
saveRDS(pred_jobs_100pct, 'pred_jobs_100pct.RDS')

#Gather predictions
pred_list <- pblapply(pred_jobs_100pct, function(p_job){
  data.table(GetPredictions(projectObject, p_job, maxWait=3600, type='probability'))
})

##########################################
# Make sub
##########################################

#Setup
skel <- fread('input/sample_submission.csv')
preds <- rbindlist(pred_list)
preds[,test_id := as.integer(d[['test_id']])]
setnames(preds, 'V1', 'is_duplicate')
setkeyv(preds, 'test_id')
setcolorder(preds, names(skel))
all.equal(skel$test_id, preds$test_id)

#Check time
ts <- as.integer(Sys.time())

#Save
fn <- paste0('output/sub_plain', ts, '.csv')
write_csv(preds, fn)
print(fn)
