library(shiny)
library(datarobot)
library(dplyr)
library(DT)
#library(plyr)
library(tidyr)
library(stringr)
library(shinythemes)
library(rjson)
library(httr)
library(ggplot2)
library(ggthemes)

as.numeric.factor <- function(x) {as.numeric(levels(x))[x]}



##Get connected - change to your credentials
userinfo <- read.csv("/Users/rajiv.shah/.logindr",stringsAsFactors = FALSE)
apitoken <- userinfo$apitoken
username <- userinfo$username
ConnectToDataRobot(endpoint ='https://app.datarobot.com/api/v2', token=apitoken)

##Project to analyze  . . . must be coded in
projectid <- "5980944cc8089175d74ac6a6"  #Adult
## Projects with derived features may cause problems for the Predict Tab
#projectid <- "5968b955c808916b2b24f999"  #Lending Club

##For Predictions
predictionServer = "https://57acb8d9c8089145a7317b1b.orm.datarobot.com"
drKey = "544ec55f-61bf-f6ee-0caf-15c7f919a45d"
maxRCcode <- 5
projectinfo <- GetProject(projectid)

##For Predictions . . .  must be coded in
predictionServer = "https://57acb8d9c8089145a7317b1b.orm.datarobot.com"
drKey = "544ec55f-61bf-f6ee-0caf-15c7f919a45d"
maxRCcode <- 5

##Initial Model for app
allModels <- GetAllModels(projectid)
modelFrame <- as.data.frame(allModels)
modelFrame <- modelFrame %>% mutate (rowid = 1:nrow(modelFrame))
metric <- modelFrame$validationMetric
bestIndex <- which.min(metric)
bestModel <- allModels[[bestIndex]]

#List of features for app (EDA page) . . . change to reactive if we need to switch projects
featurelist <- GetFeaturelist(projectid, bestModel$featurelistId)
dat <- featurelist$features
getallfeatures <- lapply(dat, function(x) {
  t <- GetFeatureInfo(projectid, x)
  t}
)
flattened <- lapply(getallfeatures, unlist)
eda2 <- as.data.frame(t(as.data.frame(flattened)))
rownames(eda2) <- c()
eda2 <- eda2 %>% dplyr::select (-projectId) %>% mutate (name = as.character(name)) %>% 
  mutate (importance = as.numeric.factor(importance)) %>% 
  arrange(desc(importance)) %>% 
  mutate (importance = round(importance,4))

