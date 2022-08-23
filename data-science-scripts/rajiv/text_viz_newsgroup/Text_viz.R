##Simple example of textviz using newsgroup dataset

##Newsgroup data
# http://qwone.com/~jason/20Newsgroups/ with only two groups (christian & atheism)
#Newgroup data for this script is at: https://drive.google.com/drive/folders/0BxMwz9lP36erdExwOEc1cjgzOWM?usp=sharing

library(datarobot)
library(dplyr)
library(htmltools)
library(pipeR)
library(colormap)
library(stringr)
library(ngram)
library(dplyr)
library(tidyr)

#source functions
source("Text_viz_fun.R")

##Get connected
userinfo <- read.csv("/Users/rajiv.shah/.logindr",stringsAsFactors = FALSE)
apitoken <- userinfo$apitoken
ConnectToDataRobot(endpoint ='https://app.datarobot.com/api/v2', token=apitoken)


#Build Model
df <- read.csv("newsgroups.csv")
project <- SetupProject(dataSource = df, 
                        projectName = "newgroups")
SetTarget(project = project, target = "class")
UpdateProject(project = project$projectId, workerCount = 10)
WaitForAutopilot(project, verbosity = 1, timeout = 999999)

#Use Existing model
projectid <- "59aee5806aaee80627461817"  #Newsgroup project for Rajiv

##############
## Grabbing Coefficients - Option A - Use a Word Cloud
##############

#In this case, I got the model id via the GUI - Look for the Eye symbol
wc <- GetWordCloud(projectid, "59aee5b18eaa70e529554a18")
coef_df <- wc %>% mutate (textname = as.character(ngram),abscoff = abs(coefficient))


##############
## Grabbing Coefficients - Option B - From a model that offers coefficients
##############
##Look for the top performing model with a Beta tag
nlpmodelparam <- GetModelParameters(projectid, "59aee5b18eaa70e529554a18")

ll <- lapply(nlpmodelparam$derivedFeatures, unlistcat)
ll2 <- ll[ ! sapply(ll, is.null) ]
coefdf <- as.data.frame(t(as.data.frame(ll2)))
coef_df <- coefdf %>% mutate(coefficient = as.numeric.factor(coefficient)) %>% 
  filter (coefficient != 0) %>% separate(derivedFeature, c("class", "textname"), "-") %>% 
  mutate (abscoff = abs(coefficient)) %>% 
  filter(abscoff > quantile(abscoff, .9)) %>% 
  mutate (coefficient = coefficient/max(coefficient),abscoff = abs(coefficient))
##Goal is dataframe that has coefficient, textname, 


##############
## Running Viz on Data
##############

## Using the GUI, i ran predictions on all the data and included columns for text and actual class
data <- read.csv("newgroups_Elastic-Net_Classifier_(L2___Binomial_Deviance)_(4)_64.04_Informative_Features.csv")
data <- data[-1]

##Subset data for a test
datasub <- data[1:10,]

##x needs to be the text field to process
x <- as.character(datasub$textname)
y <- lapply(x,ngramviz)
y <- lapply(y,function (x) HTML(x)) 

##############
## Display Viz
##############

##display in rstudio viewer window
display <- 9
header <- paste0("Row #: ",display," Actual: ",data$class[display]," Predicted: ",round (data$Cross.Validation.Prediction[display],2), " <br> " )
texttoshow <- c(header,y[[display]])
texttoshow %>>%
  HTML %>>%
  html_print

##############
## Output as HTML pages
##############

counter <- 0
test <- lapply(y, function (x) {
  counter <<- counter + 1
  filename <- paste0("output_", counter, ".html")
  save_html(x, filename, background = "white")
}
)

##############
## One function to do it all using WorldCloud
##############

##Prep - Get Word Cloud
#In this case, I got the model id via the GUI - Look for the Eye symbol
wc <- GetWordCloud(projectid, "59aee5b18eaa70e529554a18")
##Must have a dataframe with the that has a column with name textname, in this case datasub
tv_wc_htmlfile (wc,datasub)
