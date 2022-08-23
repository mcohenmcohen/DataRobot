################################################################
# Setup
################################################################
rm(list=ls(all=T))
gc(reset=T)
library(mlbench)
library(data.table)
library(datarobot)
library(pbapply)
set.seed(10)
n = 1e5
ncol = 10

x <- matrix(runif(ncol * n), ncol = ncol)
x[,4] <- x[,1] + x[,4] * .20
y <- 10 * sin(pi * x[, 1] * x[, 2])
y <- y + 20 * (x[, 3] - 0.5)^2 + 10 * x[, 4] + 5 * x[, 5]
y <- y + rnorm(n, sd = 10)
cors <- cor(x)
diag(cors) <- 0
max(cors)
# print(round(cors, 3))

jit <- jitter(x, amount = .05)
diag(cor(x, jit))
model.data <- data.table(y, x, jit)
setnames(model.data, c(
  "y",
  paste0("useful", 1:5),
  paste0("noise", 1:5),
  paste0("jitter_useful", 1:5),
  paste0("jitter_noise", 1:5)
))

filename <- '~/Downloads/Zachs_Mod_Friedman_Correlation_Jitter.csv'
fwrite(model.data, filename)

################################################################
# Run DR
################################################################

# Start project
# projectObject <- GetProject('618b40826be50e9fe4cbbf65')
projectObject = SetupProject(filename)
sink <- UpdateProject(projectObject, workerCount=25, holdoutUnlocked=TRUE)
st <- SetTarget(
  project=projectObject,
  target="y",
  targetType='Regression',
  metric='R Squared',
  partition=CreateRandomPartition(validationType='CV', holdoutPct=0, reps=5),
  smartDownsampled=FALSE,
  mode='comprehensive',
  seed=17599,
  maxWait=600)

# Function to run repo models
try_model <- function(pid, bp, scoringType='crossValidation', samplePct=NULL){
  tryCatch({
    suppressMessages({
      RequestNewModel(pid, list(
        projectId=pid$projectId,
        created=pid$created,
        projectName=pid$projectName,
        fileName=pid$fileName,
        blueprintId=bp
      ), scoringType=scoringType, samplePct=samplePct)
    })
  }, error=function(e) warning(e))
}

# Cross validate all models on the LB and all models in the repo
# Try every hour to run CV
for(i in 1:5){
  Sys.sleep(3600*1)
  models <- c(ListModels(projectObject), ListBlueprints(projectObject))
  bps <- sort(unique(sapply(models, '[[', 'blueprintId')))
  new <- pblapply(bps, function(bp){
    try_model(projectObject, bp, 'crossValidation')
    try_model(projectObject, bp, 'validation')
    Sys.sleep(0.1)
  })
}
rm(i)

# Run feature impact for top model
Sys.sleep(3600*3)
best_model <- ListModels(projectObject)[[1]]
featureImpactJobId <- RequestFeatureImpact(best_model)

# Lookit the project
ViewWebModel(best_model)
