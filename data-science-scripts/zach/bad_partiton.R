#Make a dataset
library(data.table)
library(readr)
set.seed(42)
N <- 1e4
x <- rnorm(N)
dat <- data.table(
  x = x,
  y = x * 5 + rnorm(N),
  folds = sample(1:5, N, replace=T)
)
write_csv(dat, '~/datasets/bad_partition.csv')

dat[,folds := paste0('f', folds)]
write_csv(dat, '~/datasets/fine_partition.csv')

#Make a DR Project
library(datarobot)
filename <- '~/datasets/bad_partition.csv'
ConnectToDataRobot(
  endpoint ='https://app.datarobot.com/api/v2',
  token='ZSa8G0D6TzCYUmVqDh4TB25dU0Cq8tyd')
projectObject <- SetupProject(dataSource = filename, projectName = filename)
SetTarget(
  project = projectObject,
  target = "y",
  partition=CreateUserPartition(
    validationType="CV",
    userPartitionCol="folds",
    cvHoldoutLevel="Please delete this project, I don't need it")
)
ViewWebProject(projectObject)
