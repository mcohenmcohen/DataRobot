
rm(list=ls(all=T))
gc(reset=T)
library(datarobot)

HackedCreateUserPartition <- function(
  validationType, userPartitionCol, cvHoldoutLevel = NULL,
  trainingLevel = NULL, holdoutLevel = NULL, validationLevel = NULL) {
  partition <- list(
    cvMethod = "user", validationType = validationType,
    userPartitionCol = userPartitionCol)
  if (validationType == "CV") {
    partition <- c(partition, list(cvHoldoutLevel = cvHoldoutLevel))
  }
  else if (validationType == "TVH") {
    if (is.null(trainingLevel)) {
      stop(strwrap("Parameter trainingLevel must be specified for user\n
                   partition with validationType = 'TVH'"))
    }
    else {
      partition <- c(partition, list(trainingLevel = trainingLevel))
      partition <- c(partition, list(holdoutLevel = holdoutLevel))
    }
    if (is.null(validationLevel)) {
      stop(strwrap("Parameter validationLevel must be specified for user\n
                   partition with validationType = 'TVH'"))
    }
    else {
      partition <- c(partition, list(validationLevel = validationLevel))
    }
    }
  else {
    stop(strwrap(paste("validationType", validationType,
                       "not valid for user partitions")))
  }
  class(partition) <- "partition"
  return(partition)
}

HackedSetTarget <- function(
  project,
  target,
  metric = NULL,
  weights = NULL,
  partition = NULL,
  mode = NULL,
  seed = NULL,
  positiveClass = NULL,
  blueprintThreshold = NULL,
  responseCap = NULL,
  recommenderUserId = NULL,
  recommenderItemId = NULL,
  quickrun = NULL,
  featurelistId = NULL,
  maxWait = 60
) {
  if (is.null(target)) {
    stop("No target variable specified - cannot start Autopilot")
  } else {
    if (!is.null(mode) && mode == AutopilotMode$SemiAuto) {
      Deprecated("semi mode (use auto or manual mode instead)",
                 "2.3", "3.0")
    }
    projectId <- datarobot:::ValidateProject(project)
    routeString <- datarobot:::UrlJoin("projects", projectId, "aim")
    pStat <- datarobot:::GetProjectStatus(projectId)
    stage <- as.character(pStat[which(names(pStat) == "stage")])
    if (stage != "aim") {
      errorMsg <- paste("Autopilot stage is", stage, "but it must be 'aim' to set the target and start a new project")
      stop(strwrap(errorMsg))
    }

    if (is.numeric(mode)) {
      Deprecated("Numeric modes (use e.g. AutopilotMode$FullAuto instead)",
                 "2.1", "3.0")
    }
    bodyList = list(
      target = target,
      metric = metric,
      weights = weights,
      mode = mode,
      seed = seed,
      positiveClass = positiveClass,
      blueprintThreshold = blueprintThreshold,
      responseCap = responseCap,
      recommenderUserId = recommenderUserId,
      recommenderItemId = recommenderItemId,
      quickrun = quickrun,
      featurelistId = featurelistId
    )
    bodyList <- c(bodyList, partition)
    if (length(bodyList$partitionKeyCols) == 0) {
      body <- jsonlite::unbox(as.vector(bodyList))
    }
    else {
      body <- datarobot:::FormatMixedList(bodyList, specialCase = "partitionKeyCols")
    }
    response <- datarobot:::DataRobotPATCH(
      routeString, addUrl = TRUE,
      body = body, returnRawResponse = TRUE, encode = "json")
    datarobot:::WaitForAsyncReturn(
      httr::headers(response)$location,
      addUrl = FALSE, maxWait = maxWait, failureStatuses = "ERROR")
    message("Autopilot started")
  }
}

ConnectToDataRobot(
  endpoint ='https://app.datarobot.com/api/v2',
  token='ZSa8G0D6TzCYUmVqDh4TB25dU0Cq8tyd')

projectObject <- SetupProject(dataSource = '~/datasets/10kdiabetes-manual-folds.csv', projectName = '10kdb')
st <- SetTarget(
  project = projectObject,
  target = "loss",
  partition=HackedCreateUserPartition(
    validationType='CV',
    userPartitionCol='fold'
  ),
  metric='RMSLE',
  quickrun=FALSE,
  maxWait=600)




bodyList = list(
  target = "loss",
  metric = "RMSLE",
  cvMethod = "user",
  validationType = "CV",
  userPartitionCol = "fold",
  cvHoldoutLevel = NA
)
jsonlite::toJSON(bodyList, auto_unbox = TRUE)

