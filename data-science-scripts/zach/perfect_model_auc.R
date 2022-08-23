# Setup
library(data.table)
library(datarobot)
library(caTools)
library(Metrics)
library(ggplot2)
library(ggthemes)

# Load the customer's dataset + trim down to just the features we need
goodbad = fread('~/datasets/customer/Good_vs_Bad.csv')
goodbad[,id := 1:.N]
FI = fread('~/workspace/data-science-scripts/zach/feature_impacts.csv')
keep <- c('id', 'status', FI[['Feature Name']])
keep <- setdiff(keep, c('dateentered (Day of Month)'))
goodbad <- goodbad[,keep,with=F]

# Run the autopilot
ConnectToDataRobot()
pid <- SetupProject(dataSource = goodbad, projectName = 'goodbad', maxWait=3600)
up <- UpdateProject(pid, workerCount = 20, holdoutUnlocked = TRUE)
st <- SetTarget(
  project = pid, target = "status",
  metric='AUC',
  maxWait=600)
pid = GetProject('5a4d3fc710875908c8af162c')
ViewWebProject(pid)

#Manually run/tune some models here, to get no feature impact+feature impact
#Use RF blueprints, manually train at 100%, and then do 2 models:
#1. max_depth=None, max_leaf_nodes=2000, n_estimators=500 (no impact)
#2. max_depth=10, max_leaf_nodes=None, n_estimators=100 (has impact)

# Load the 2 models
no_impact_lid = GetModelObject(pid, '5a4d44a46d022d6ffc599e04')
impact_lid = GetModelObject(pid, '5a4d47fc6d022d720edbfc77')

ViewWebModel(no_impact_lid)
ViewWebModel(impact_lid)

no_impact_feature_impact = datarobot::GetFeatureImpactForModel(no_impact_lid)
impact_raw_feature_impact = datarobot::GetFeatureImpactForModel(impact_lid)

# Make predictions
pred_data = UploadPredictionDataset(pid, goodbad, maxWait=3600)
no_impact_pred_job = RequestPredictionsForDataset(pid, no_impact_lid$modelId, pred_data$id)
impact_pred_job = RequestPredictionsForDataset(pid, impact_lid$modelId, pred_data$id)
Sys.sleep(10)

# Get prediction
goodbad[,no_impact_preds := GetPredictions(pid, no_impact_pred_job, type='probability')]
goodbad[,impact_preds := GetPredictions(pid, impact_pred_job, type='probability')]

# AUC
no_impact_auc = goodbad[,colAUC(no_impact_preds, status)]
impact_auc = perfect[,colAUC(impact_preds, status)]

# logloss
no_impact_ll = goodbad[,logLoss(status=='good', no_impact_preds)]
impact_ll = perfect[,logLoss(status=='good', impact_preds)]

# Spread
find_spread <- function(x, g){
  min(x[g=='good']) - max(x[g=='bad'])
}
no_impact_spread = goodbad[,find_spread(no_impact_preds, status)]
impact_spread = goodbad[,find_spread(impact_preds, status)]

# Plots
ggplot(good, aes(x = no_impact_preds)) +
  geom_density(aes(fill = status), alpha = 0.5) +
  ggtitle(
    paste(
      "No Impacts Model Separation:\n    In-Sample AUC =", no_impact_auc, "\n",
      "   In-Sample logloss =", no_impact_ll, "\n",
      "   good-bad prediction gap = ", no_impact_spread
    )) +
  theme_bw() + theme_tufte() + theme(legend.position='top')

ggplot(good, aes(x = impact_preds)) +
  geom_density(aes(fill = status), alpha = 0.5) +
  ggtitle(
    paste(
      "Has Impacts Model Separation:\n    In-Sample AUC =", impact_auc, "\n",
      "   In-Sample logloss =", impact_ll, "\n",
      "   good-bad prediction gap = ", impact_spread
    )) +
  theme_bw() + theme_tufte() + theme(legend.position='top')

