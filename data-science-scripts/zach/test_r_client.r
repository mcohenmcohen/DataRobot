# NOTE: you need to install the dev version of the client:
# devtools::install("~/workspace/public_api_R_client/")

# Connect to datarobot
library(datarobot)
token <- 'NjA4OTI4N2NhNWIyM2I2YzdiOGMyZTFjOnN0YWdpbmcudGVzdC51c2VyK21iLXRlc3QtNjA4OTI4NTIwNDZlMjM2OWNhOGMxZWI0QGRhdGFyb2JvdC5jb20='
endpoint <- 'http://owens-world.hq.datarobot.com/api/v2'
ConnectToDataRobot(endpoint = endpoint, token = token)

# Load the csv of data
data <- read.csv("~/workspace/data-science-scripts/zach/removed.csv", header = TRUE)

# Add error handling to GetAnomalyAssessmentExplanations / GetAnomalyAssessmentPredictionsPreview
# https://stackoverflow.com/a/12195574/345660
GetAnomalyAssessmentExplanations_HandleErrors <- function(...){
  tryCatch(
    GetAnomalyAssessmentExplanations(...),
    error=function(cond) {
      message('GetAnomalyAssessmentExplanations Failed')
      message("Here's the original error message:")
      message(cond)
      # Choose a return value in case of error
      return(NA)
    }
  )
}

GetAnomalyAssessmentPredictionsPreview_HandleErrors  <- function(...){
  tryCatch(
    GetAnomalyAssessmentPredictionsPreview(...),
    error=function(cond) {
      message('GetAnomalyAssessmentPredictionsPreview Failed')
      message("Here's the original error message:")
      message(cond)
      # Choose a return value in case of error
      return(NA)
    }
  )
}

# Loop over the rows in the csv
for(row_number in 1:nrow(data)){

  row <- data[row_number,]
  print(paste0("project id:", row["pid"]))

  startDate <- row[["date"]]
  PID <- row[["pid"]]
  LID <- row[["lid"]]
  BACKTEST <- row[["backtest"]]

  # Determine single or multi series
  if (row["series"] != "None") {
    print("data is multi series")
    seriesId <- row["series"]
  } else if(row["series"] == "None") {
    print("data is single series")
    seriesId <- NULL
  } else{
    stop("Series is not multi series or single series.  Something went wrong!")
  }

  # Pull the anomaly assessment records
  records <- ListAnomalyAssessmentRecords(PID, LID, backtest=BACKTEST, seriesId=seriesId)

  # Now loop over the records
  for(r in records){
    record_id <- record[["recordId"]]
    explanations <- GetAnomalyAssessmentExplanations_HandleErrors(PID, recordId=record_id, startDate=startDate, pointsCount=5000)
    preview <- GetAnomalyAssessmentPredictionsPreview_HandleErrors(PID, recordId=record_id)
  }
}
