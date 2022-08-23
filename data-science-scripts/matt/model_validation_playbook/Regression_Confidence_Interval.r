#-----------------------------------------------------------------------------
# Confidence Interval Estimation Example-DataRobot API
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# Load the required packages
#-----------------------------------------------------------------------------
InstalledPackage <- function(package) 
{
  available <- suppressMessages(suppressWarnings(sapply(package, 
                                                        require, quietly = TRUE, 
                                                        character.only = TRUE, 
                                                        warn.conflicts = FALSE)))
  missing <- package[!available]
  if (length(missing) > 0) return(FALSE)
  return(TRUE)
}

CRANChoosen <- function()
{
  return(getOption("repos")["CRAN"] != "@CRAN@")
}

UsePackage <- function(package, defaultCRANmirror = "http://cran.at.r-project.org") 
{
  if(!InstalledPackage(package))
  {
    if(!CRANChoosen())
    {       
      chooseCRANmirror()
      if(!CRANChoosen())
      {
        options(repos = c(CRAN = defaultCRANmirror))
      }
    }
    
    suppressMessages(suppressWarnings(install.packages(package)))
    if(!InstalledPackage(package)) return(FALSE)
  }
  return(TRUE)
}

libraries <- c("datarobot","caret", "purrr","MASS","boot","dplyr")
for(library in libraries) 
{ 
  if(!UsePackage(library))
  {
    stop("Error!", library)
  }
}
#-----------------------------------------------------------------------------
# DataRobot API Connections
#-----------------------------------------------------------------------------
ConnectToDataRobot(endpoint ='https://app.datarobot.com/api/v2', token='2Gq3kdCLTw2vCCOZHthw5BoYGATgJKKT')
#using the Cars dataset from Caret for illustration
data("cars")
str(cars)

##dats from census

##append the columns to cars
#-----------------------------------------------------------------------------
# Original Coefficient Estimates-Set up the Project in DataRobot, Extract Model Results
#-----------------------------------------------------------------------------
projectObject <- SetupProject(dataSource = cars, projectName = "CarsVignetteProject")
SetTarget(project = projectObject, target = "Price")
WaitForAutopilot(project = projectObject)
listOfModels <- GetAllModels(projectObject)
modelFrame <- as.data.frame(listOfModels)
ga2m<-modelFrame[modelFrame$modelType == "Generalized Additive Model (Gamma Loss)", ]
df<-as.list(GetModelParameters(projectObject$projectId,ga2m$modelId))
df<-flatten(df)
coefficient <- sapply(df, function(x)x["coefficient"])
#ElasticNet<-modelFrame[modelFrame$modelType == "Elastic-Net Regressor (L2 / Gamma Deviance) with Binned numeric features", ]
df<-as.list(GetModelParameters(projectObject$projectId,ElasticNet$modelId))
df<-flatten(df)
coefficient <- sapply(df, function(x)x["coefficient"])
#-----------------------------------------------------------------------------
## A helper function that tests whether an object is either NULL _or_ 
## a list of NULLs
is.NullOb <- function(x) is.null(x) | all(sapply(x, is.null))
## Recursively step down into list, removing all such objects 
rmNullObs <- function(x) {
  x <- Filter(Negate(is.NullOb), x)
  lapply(x, function(x) if (is.list(x)) rmNullObs(x) else x)
}
#-----------------------------------------------------------------------------
coefficient<-rmNullObs(coefficient)
coefficient<-as.data.frame(coefficient)
coefficient<-do.call(rbind.data.frame, coefficient)
colnames(coefficient)<-"values"
original.estimates<-t(coefficient)
#-----------------------------------------------------------------------------
# Set the number of replications
#-----------------------------------------------------------------------------
n.sim <- 25 #this is a user-defined value
#-----------------------------------------------------------------------------
# The loop
#-----------------------------------------------------------------------------
set.seed(123)
coefficient_list<-list()
for(i in 1:n.sim) {
  #-----------------------------------------------------------------------------
  # Draw the observations WITH replacement
  #-----------------------------------------------------------------------------
  data.new <- cars[sample(1:dim(cars)[1], dim(cars)[1], replace=TRUE),]
  #-----------------------------------------------------------------------------
  # Set up the Project in DataRobot, Extract Model Results
  #-----------------------------------------------------------------------------
  projectObject <- SetupProject(dataSource = data.new, projectName = "CarsVignetteProject")
  SetTarget(project = projectObject, target = "Price")
  WaitForAutopilot(project = projectObject)
  listOfModels <- GetAllModels(projectObject)
  modelFrame <- as.data.frame(listOfModels)
  ElasticNet<-modelFrame[modelFrame$modelType == "Elastic-Net Regressor (L2 / Gamma Deviance) with Binned numeric features", ]
  df<-as.list(GetModelParameters(projectObject$projectId,ElasticNet$modelId))
  df<-flatten(df)
  coefficient <- sapply(df, function(x)x["coefficient"])
  coefficient<-rmNullObs(coefficient)
  coefficient<-as.data.frame(coefficient)
  coefficient<-do.call(rbind.data.frame, coefficient)
  colnames(coefficient)<-"values"
  #-----------------------------------------------------------------------------
  # Store the results
  #-----------------------------------------------------------------------------
  coefficient_list[[i]] <- t(coefficient)
}
coefficient_final <- do.call(rbind, coefficient_list)
#-----------------------------------------------------------------------------
# Save the means, medians and SDs of the bootstrapped statistics
#-----------------------------------------------------------------------------
boot.means <- colMeans(coefficient_final, na.rm=T)
boot.medians <- apply(coefficient_final,2,median, na.rm=T)
boot.sds <- apply(coefficient_final,2,sd, na.rm=T)
#-----------------------------------------------------------------------------
# The bootstrap bias is the difference between the mean bootstrap estimates
# and the original estimates
#-----------------------------------------------------------------------------
boot.bias <- colMeans(coefficient_final, na.rm=T) - original.estimates
#-----------------------------------------------------------------------------
# Bootstrap CIs based on the empirical quantiles
#-----------------------------------------------------------------------------
conf.mat <- matrix(apply(coefficient_final, 2 ,quantile, c(0.025, 0.975), na.rm=T),
                   ncol=2, byrow=TRUE)
colnames(conf.mat) <- c("95%-CI Lower", "95%-CI Upper")
#-----------------------------------------------------------------------------
# Set up summary data frame
#-----------------------------------------------------------------------------
summary.frame <- data.frame(mean=boot.means, median=boot.medians,
                            sd=boot.sds, "CI_lower"=conf.mat[,1], "CI_upper"=conf.mat[,2])
summary.frame
