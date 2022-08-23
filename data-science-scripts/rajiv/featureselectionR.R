
library(datarobot)
library(dplyr)
library(stringr)
library(ggplot2)

userinfo <- read.csv("/Users/rajiv.shah/.logindr",stringsAsFactors = FALSE)
apitoken <- userinfo$apitoken

ConnectToDataRobot(endpoint ='https://app.datarobot.com/api/v2', token=apitoken)

projectid <- "5a9abf375a3323254f7f9d33"

##Get the list of models you want to work with
models <- ListModels(projectid)
modelsdf<- as.data.frame(models)
bestmodels <- modelsdf %>% mutate(rowname = 1:n()) %>% 
  filter (featurelistName == "Informative Features",samplePct==64,!str_detect(expandedModel, 'Blender')) 

##Collect all the feature impact info
all_impact<- NULL
for(i in 1:nrow(bestmodels)) {  
    featureImpactJobId <- NULL
    modelid <- models[[bestmodels[i,]$rowname]]
    tryCatch({
      featureImpactJobId <- RequestFeatureImpact(modelid)
      featureImpact <- GetFeatureImpactForJobId(projectid, featureImpactJobId,maxWait = 10000)
      }, 
      error = function(e) {      
        return({
          paste( "Rerunning:", conditionMessage(e))
          featureImpact <<-  GetFeatureImpactForModel(modelid)
        });
      }
    )
    featureImpact <- featureImpact %>% 
      arrange(desc(impactUnnormalized)) %>% 
      mutate (rankn = 1:n())  ##Using ranked impact
    all_impact <- rbind(all_impact,featureImpact)
  }


##Plot the features 
p <- ggplot(all_impact, aes(x=reorder(featureName, -rankn, FUN=median), y=rankn))
p + geom_boxplot() + coord_flip()

## Process Feature impact
ranked_impact <- all_impact %>% group_by(featureName) %>% 
    summarise(impact = mean(rankn),
              impact_sd = sd(rankn)) %>% 
    arrange(impact)

topfeatures <- pull(ranked_impact,featureName)
No_of_features_to_select <- 4

# Run AutoPilot on new Top Features
listname = paste0("TopFI_", No_of_features_to_select)
Feature_id_percent_rank = CreateFeaturelist(projectid, listName= listname , featureNames = topfeatures[1:No_of_features_to_select])$featurelistId
StartNewAutoPilot(projectid,featurelistId = Feature_id_percent_rank)
WaitForAutopilot(projectid, checkInterval = 20, timeout = NULL, verbosity = 1)
