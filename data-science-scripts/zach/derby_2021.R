gc(reset=T)
rm(list=ls(all=T))
library(data.table)
library(jsonlite)

############################################################################
# Load data and make ordinal target
############################################################################
# NOTE: open the file first in excel and convert all percent/$ to pure num

clean_data <- function(filename){
  x <- fread(filename)
  x[,`PP cat` := paste0('p', `Post Position`)]

  x[,sp_odds_norm := 1/(1+`Starting Price`)]
  x[,sp_odds_norm := sp_odds_norm/sum(sp_odds_norm, na.rm=T), by='Year']
  x[,sp_odds_norm_inv := qlogis(sp_odds_norm)]

  x[,prime_power_norm := `Prime Power` / sum(`Prime Power`, na.rm=T), by='Year']
  x[,rcg_spd_avg_norm := `Rcg Spd Avg` / sum(`Rcg Spd Avg`, na.rm=T), by='Year']
  x[,best_e1_norm := `Best E1` / sum(`Best E1`, na.rm=T), by='Year']
  x[,best_e1_norm := `Best E2` / sum(`Best E2`, na.rm=T), by='Year']
  x[,sp1_norm := Sp1 / sum(Sp1, na.rm=T), by='Year']

  return(x)
}

# Training data
dat_train <- clean_data('~/Downloads/Kentucky Derby 2021 Historical Data.csv')
dat_train[,finish_ord := sapply(`Finish Position`, function(x){
  toJSON(as.character(seq(x, 20)))
})]
dat_train[,finish_ord := gsub('"0",', '', finish_ord, fixed=T)]
dat_train[,finish_ord := gsub('"0"', '', finish_ord, fixed=T)]
dat_train[,`Finish Position` := NULL]

# Test data
dat_test <- clean_data('~/Downloads/Kentucky Derby 2021 Predictor Data.csv')
dat_test[,`Finish Position` := NULL]

# Check
stopifnot(setdiff(names(dat_train), names(dat_test)) == 'finish_ord')
stopifnot(setdiff(names(dat_test), names(dat_train)) == character(0))
for(var in names(dat_test)){
  stopifnot(class(dat_train[[var]]) == class(dat_test[[var]]))
}

############################################################################
# Run DataRobot
############################################################################

# Patch the api client to allow Multilabel
# Note you need the multilabel API client
TargetType <- list(
  Binary = "Binary",
  Multiclass = "Multiclass",
  Regression = "Regression",
  Multilabel = "Multilabel"
  )
assignInNamespace("TargetType", TargetType, ns="datarobot")

# Upload dataset
projectObject <- SetupProject(dataSource = dat_train, projectName = 'Derby 2021')
up <- UpdateProject(projectObject, workerCount = 25, holdoutUnlocked = TRUE)

# Set target
st <- SetTarget(
  project = projectObject,
  target = "finish_ord",
  partition=CreateUserPartition(
    validationType='CV',
    userPartitionCol='Year'
  ),
  metric='LogLoss',
  targetType='Multilabel',
  mode='comprehensive',
  maxWait=600)

print(projectObject['projectId'])

# NOTE: be sure to manually run every model with all CV folds, using repo
# NOTE: Once the autopilot is done, also manually run all models on the reduced FL

# Upload prediction file
testfile <- UploadPredictionDataset(projectObject, dat_test, maxWait=3600)

############################################################################
# Select best model
############################################################################
#Sys.sleep(30*60)

# Find the 100 model(s)
models <- ListModels(projectObject)
samplePct <- sapply(models, '[[', 'samplePct')
pred <- models[samplePct == 100]

# If more than 1, select by (stacked) CV score
loss <- sapply(pred, function(x) x$metrics[['LogLoss']]$crossValidation)
best <- pred[order(loss)][[1]]

# Predict
pred_raw <- Predict(best, predictionDataset = testfile, maxWait = 3600, type = "raw")

# Format preds
pred <- lapply(1:20, function(i){
  data.table(
    row=pred_raw$rowId[i]+1,
    label=pred_raw$predictionValues[[i]][['label']],
    value=1-pred_raw$predictionValue[[i]][['value']]
  )
})
pred <- rbindlist(pred)
pred[, label := as.integer(label)]
pred[,name := dat_test[['Horse Name']][row]]
pred <- dcast.data.table(pred, name + row ~ label)

# Normalize preds
for(i in as.character(19:1)){
  setorderv(pred, i, order=-1)
}
pred[,sp_odds_norm := dat_test[['sp_odds_norm']][row]]

# Drop some columns
pred[, row := NULL]
pred[, `0` := NULL]

# Show "good bets"
pred[,bet_ratio := `1` / sp_odds_norm]
#setorderv(pred, 'bet_ratio', order=-1)
pred
sum(unlist(pred[name=='Essential Quality',][,as.character(1:19),with=F]))
sum(unlist(pred[name=='O Besos',][,as.character(1:19),with=F]))

# Save
pred_out <- copy(pred)
pred_out[,bet_ratio := NULL]
pred_out[,sp_odds_norm := round(sp_odds_norm, 4)]
fwrite(pred_out, '~/workspace/data-science-scripts/zach/derby_preds.csv')

############################################################################
# Plot
############################################################################

plot_horse <- function(horse){
  odds <- pred[name == horse,]
  odds <- sapply(1:19, function(i) odds[[as.character(i)]])
  plot(odds, main=horse)
}
plot_horse('Essential Quality')

lines_horse <- function(horse){
  odds <- pred[name == horse,]
  odds <- sapply(1:19, function(i) odds[[as.character(i)]])
  lines(odds)
}

for(h in pred[['name']]){
  lines_horse(h)
}
