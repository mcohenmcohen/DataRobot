
# Set your filepath -------------------------------------------------------

my_filepath <- "/Users/taylor.larkin/Documents/DRU R Class/New Content"

# Loading packages --------------------------------------------------------

library(datarobot)
library(dplyr)
library(reshape2)
library(MLmetrics)
library(ggplot2)

# Loading functions for class ---------------------------------------------

setwd(my_filepath)
source("functions_for_class.R")

# Connecting to DataRobot -------------------------------------------------

ConnectToDataRobot(configPath = "drconfig.yaml")

# Data preparation --------------------------------------------------------

# Downloading dataset to working directory
download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
              destfile = "wine_data.csv", method = "libcurl")

# Loading dataset
wine_data <- read.csv("wine_data.csv", sep = ";")

# Checks
head(wine_data)
hist(wine_data$quality)
table(wine_data$quality)

# Good Wine?
wine_data$wine_is_good <- as.factor(ifelse(wine_data$quality >= 7, "good", "bad"))
prop.table(table(wine_data$wine_is_good))

# Seperate into training and testing
set.seed(10)
indices <- createTrainTest(target = wine_data$wine_is_good, sample_size = 0.9)
training <- wine_data[indices,]
testing <- wine_data[-indices,]

# Start of Project --------------------------------------------------------

# Creating and starting project in the cloud
project_object <- StartProject(dataSource = training, projectName = "Wine Quality - Basics",
                               target = "wine_is_good", mode = "quick", positiveClass = "good")

# Add max workers
UpdateProject(project_object, workerCount = -1)

# Creates verbose indicating how many are in progress, in queue with wait times
WaitForAutopilot(project_object)

# See project info
project_info <- GetProject(project_object)

# Collects all the models from project as a sorted list object
all_models <- ListModels(project_object)

# Creates a nice data.frame of results
as.data.frame(all_models)

# Grabbing the recommended model
best_model <- GetRecommendedModel(project_object, type = "Recommended for Deployment")

# Description of best model
model_info <- GetBlueprintDocumentation(project_object, blueprintId = best_model$blueprintId)
str(model_info)

# Getting Predictions -----------------------------------------------------

# Uploading the testing dataset
scoring <- UploadPredictionDataset(project_object, dataSource = testing)

# Requesting prediction
predict_job_id <- RequestPredictionsForDataset(project_object, 
                                               modelId = best_model$modelId, 
                                               datasetId = scoring$id)

# Grabbing predictions
predictions_prob <- GetPredictions(project_object, 
                                   predictJobId = predict_job_id, 
                                   type = "probability")

# Output
head(data.frame(True_Class = testing$wine_is_good, Probability = predictions_prob))

# Model Insights ----------------------------------------------------------

# Get the ROC curve
roc_data <- GetRocCurve(best_model, source = DataPartition$CROSSVALIDATION)

ggplot(roc_data$rocPoints, aes(x = falsePositiveRate, y = truePositiveRate)) + 
  geom_point(color = "green") + xlab("False Positive Rate (Fallout)") + ylab("True Positive Rate (Sensitivity)") + 
  theme_dark() + 
  annotate("text", x = .75, y = .25, color = "white", 
           label = paste("AUC =", round(Area_Under_Curve(roc_data$rocPoints$falsePositiveRate, 
                                                         roc_data$rocPoints$truePositiveRate), 4)))

# Get Precision-Recall curve
ggplot(roc_data$rocPoints, aes(x = truePositiveRate, y = positivePredictiveValue)) +
  geom_point(color = "green") + xlab("Recall") + ylab("Precision") + 
  theme_dark() +
  annotate("text", x = .25, y = .25, color = "white",
           label = paste("PRAUC =", round(Area_Under_Curve(roc_data$rocPoints$truePositiveRate, 
                                                           roc_data$rocPoints$positivePredictiveValue), 4)))

# Get lift chart
lift_data <- GetLiftChart(best_model, source = DataPartition$CROSSVALIDATION)

# Create means and standard deviations
lift_data$index <- rep(1:10, each = 6) # into deciles
lift_data_predicted <- lift_data %>%
  group_by(index) %>%
  summarise(group = "predicted",
            value = mean(predicted),
            sd = sd(predicted))
lift_data_actual <- lift_data %>%
  group_by(index) %>%
  summarise(group = "actual",
            value = mean(actual),
            sd = sd(actual))

# Gather them and plot
plot_data <- bind_rows(lift_data_predicted, lift_data_actual)
ggplot(plot_data, aes(x = index, y = value, colour = group, group = group)) +
  geom_errorbar(aes(ymin = value-sd, ymax = value+sd), width=.1, position = position_dodge(0.1)) +
  geom_line(position = position_dodge(0.1)) +
  geom_point(position = position_dodge(0.1), size = 3, shape = 21, fill="white") +
  xlab("Bins based on predicted value") +
  ylab("Average target value") +
  scale_colour_hue(guide = guide_legend(title = NULL),
                   breaks=c("predicted", "actual"),
                   labels=c("Predicted", "Actual"), l = 40) +
  ggtitle("Lift chart with standard deviations") +
  scale_x_continuous(breaks=1:10) + 
  theme_bw() +
  theme(legend.justification = c(-.25,.75),
        legend.position = c(0,.75),
        legend.background = element_blank())

# Get feature impact - no need to request since recommended has already been calculated
feature_impact <- GetFeatureImpactForModel(best_model)

ggplot(data = feature_impact, aes(x = reorder(featureName, impactNormalized), y = impactNormalized)) + 
  geom_bar(stat = "identity") + coord_flip() + ylab("Effect") + xlab("") +
  scale_y_continuous(labels = function(x){ paste0(x*100, "%") })

# Prediction Explanations -------------------------------------------------

# Computes PEs on small sample of data (prereq for subsequent jobs)
# Note prediction explanation = reason code
pe_job_id <- RequestReasonCodesInitialization(best_model)
pe_job_id_init <- GetReasonCodesInitializationFromJobId(project_object, pe_job_id)

# Computes top 3 prediction explanations on testing dataset
pe_request <- RequestReasonCodes(best_model, scoring$id, maxCodes = 3)
pe_request_metadata <- GetReasonCodesMetadataFromJobId(project_object, pe_request)
pe_frame <- GetAllReasonCodesRowsAsDataFrame(project_object, pe_request_metadata$id)

head(pe_frame)

# Feature that appeared in top explanation the most across testing dataset?
sort(table(pe_frame$reason1FeatureName), decreasing = TRUE)

# Plot distribution of alcohol for good and bad wines
boxplot(alcohol ~ wine_is_good, data = testing)
title("Alcohol % on Testing Dataset")

