
require(data.table)
require('datarobot')
require('plyr')

##---******--Define your variables here--******

## Set path to data file
mpath = '/Users/swapnil.awasthi/Downloads/datasets/DR_Demo_Car_Insurance_Fraud.csv' #File path

# Enter column names here that you want to drop, it will be added to droplist
droplist <- c('LOCALITY', 'REGION')

# Enter model that you required 
model.required <- 'Double Median Absolute Deviation'
#model.required <- 'Mahalanobis'

car_data = fread(mpath, header = TRUE)
#Create a random dummy binary variable for target -let's name it DUMMY
car_data$DUMMY <- sample(2, size = nrow(car_data), replace = TRUE)
#Convert it to dataframe
car_data <- as.data.frame(car_data)

#Drop any undesired features - example LOCALITY, REGION here
cardata_withdropped <- car_data[, !(names(car_data)) %in% droplist]

# *****------Start of Project-------*****
#connect to DataRobot - drconfig will have your login credentials
ConnectToDataRobot(configPath = "~/Downloads/AdvanceR-DR/.config/datarobot/drconfig.yaml")

# Creating project in cloud
project <- SetupProject(dataSource = cardata_withdropped, projectName = "Anomaly_project", maxWait = 60 * 60)

# Sets target feature and kicks off modeling
SetTarget(project, target = "DUMMY", mode = "manual")

# Add more workers and set holdoutUnlocked = TRUE #added logic to rename project as project ID
UpdateProject(project, workerCount = 20, holdoutUnlocked = TRUE, newProjectName = project$projectId)

#Get list of BPs
bp_available <- ListBlueprints(project)
#dataframe
#bp_list_df <- as.data.frame(ListBlueprints(project.object))

#Get anomaly blueprint that you defined above
bp_available_anomaly <- Filter(function(b) grepl(model.required, b$modelType), bp_available)

#Run models for all the BPs filtered
for (blueprint in bp_available_anomaly) {
  try(RequestNewModel(project$projectId, blueprint, samplePct = 100))
}

# Collects all the models from project as a sorted list object
listofmodels <- ListModels(project)

# Uploading a dataset (simply using the same one for illustration)
scoring_dataset <- UploadPredictionDataset(project, cardata_withdropped)

# Requesting prediction --
model.id <- listofmodels[[1]]$modelId
predict.job.id <- RequestPredictionsForDataset(project = project, 
                                               modelId = model.id ,
                                               datasetId = scoring_dataset$id)

## Get predictions/Download predictions
getpredictions <- GetPredictions(project = project, 
                              predictJobId = predict.job.id, 
                              type = "probability")
# Store predictions(anomaly scores) as a dataframe and bind it with ID
getpredictions.df <- as.data.frame(getpredictions)
getpredictions.df.withrowID <- cbind(ID = cardata_withdropped$ID, getpredictions.df)

# Merge it back to the original dataset
merged.dataset <- merge(cardata_withdropped, getpredictions.df.withrowID, by=c("ID"))

# Upload new data for new project
# Creating project
project.2 <- SetupProject(dataSource = merged.dataset, projectName = "Anomaly_project_pass2", maxWait = 60 * 60)

# Sets target feature and kicks off modeling
# Target is anomaly score
SetTarget(project.2, target = "getpredictions", mode = "manual")

# Add more workers and set holdoutUnlocked = TRUE ##added logic to rename project as project ID
UpdateProject(project.2, workerCount = 20, holdoutUnlocked = TRUE, newProjectName = project.2$projectId)

for (blueprint in bp_available_anomaly) {
  try(RequestNewModel(project.2$projectId, blueprint, samplePct = 100))
}

#Run Feature Impact
listofmodels.2 <- ListModels(project.2)[[1]]

featureImpactJobId <- RequestFeatureImpact(listofmodels.2)
featureImpact <- GetFeatureImpactForJobId(listofmodels.2, jobId = featureImpactJobId)

#Run Prediction Explanations on 100% of the data (uploaded)
reasonCodeJobID <- RequestReasonCodesInitialization(listofmodels.2)
reasonCodeJobIDInitialization <- GetReasonCodesInitializationFromJobId(project.2,reasonCodeJobID)

# Uploading a dataset
scoring_dataset.2 <- UploadPredictionDataset(project.2, cardata_withdropped)
model.id.2 <- ListModels(project.2)[[1]]$modelId

# Request prediction for the uploaded data
predict.job.id.2 <- RequestPredictionsForDataset(project = project.2, 
                                               modelId = model.id.2 ,
                                               datasetId = scoring_dataset.2$id)
## Get predictions/Download predictions
getpredictions.2 <- GetPredictions(project = project.2, 
                                 predictJobId = predict.job.id.2, 
                                 type = "probability")

reasonCodeRequest <- RequestReasonCodes(listofmodels.2, scoring_dataset.2$id, maxCodes = 10)
reasonCodeRequestMetaData <- GetReasonCodesMetadataFromJobId(project.2, reasonCodeRequest, maxWait = 1800)
reasonCodeMetadata <- GetReasonCodesMetadata(project.2, reasonCodeRequestMetaData$id)
reasonCodeAsDataFrame <- GetAllReasonCodesRowsAsDataFrame(project.2,reasonCodeRequestMetaData$id)

#--- END OF CODE---#
