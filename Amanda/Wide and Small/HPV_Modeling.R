
library(datarobot)
train <- read.csv("HPV_datarobot.csv")

No_of_features_to_select = 50
no_of_models = 5
universal_impact <- NULL

for(i in 1:no_of_models) {
     seed_no = 369 + i
     project_name = paste0('HPV example',i)
     MV_project <-SetupProject(dataSource=train, projectName = project_name, maxWait =1000)
     cv_partition <- CreateRandomPartition('CV', holdoutPct = 10, reps = 10)
     SetTarget(MV_project, target='HPV', seed=seed_no, metric="AUC", partition = cv_partition, maxWait = 1000)
     UpdateProject(MV_project, workerCount = 5)
     WaitForAutopilot(MV_project, checkInterval = 20, timeout = NULL, verbosity = 1)
     UpdateProject(MV_project, holdoutUnlocked=TRUE)
     models <- GetAllModels(MV_project)
     all_impact <- NULL
     for(i in 1:length(models)) {
	featureImpactJobId <- RequestFeatureImpact(models[[i]])
	featureImpact <- GetFeatureImpactForJobId(MV_project, featureImpactJobId, maxWait = 10000)
	featureImpact$modelId <- models[[i]]$modelId
	all_impact <- rbind(all_impact, featureImpact)
     }
     universal_impact <- rbind(universal_impact, all_impact)
}

## Process Feature impact

ranked_impact <- all %>%
          arrange(modelId, -impactNormalized) %>%
          group_by(modelId) %>%
          mutate(model_rank=row_number())

ranked_impact <- as.data.frame(ranked_impact)
by_feature_rank_sum <- aggregate(ranked_impact$model_rank, by=list(ranked_impact$featureName),FUN=sum)
by_feature_rank_sum <- by_feature_rank_sum[order(by_feature_rank_sum$x),]
percent_rank <- by_feature_rank_sum[1: No_of_features_to_select,"Group.1"]

# Run AutoPilot on universal feature impact
listname = paste0("Universal_", No_of_features_to_select)
Feature_id_percent_rank = CreateFeaturelist(MV_project, listName= "Universal 2" , featureNames = percent_rank)$featurelistId
StartNewAutoPilot(MV_project,featurelistId = Feature_id_percent_rank)
WaitForAutopilot(MV_project, checkInterval = 20, timeout = NULL, verbosity = 1)
