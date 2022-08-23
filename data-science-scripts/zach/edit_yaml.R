
#Load Data
library(yaml)
x <- yaml.load_file(
  '~/workspace/mbtest-datasets/mbtest_datasets/data/Variety/mbtest_current_with_preds.yaml',
  handlers=list(
    "bool#y"=identity,
    "bool#yes"=identity,
    "bool#T"=identity,
    "bool#TRUE"=identity,
    "bool#1"=identity
  )
)

#Remove logloss (WHY DOES THIS TAKE 2 PASSES!)
metrics <- unlist(sapply(x, '[[', 'metric'))
table(metrics)
keep <- !(metrics %in% c('LogLoss', 'LogLoss', 'AUC', 'Weighted LogLoss'))
length(x)
x <- x[keep]
length(x)
metrics <- unlist(sapply(x, '[[', 'metric'))
table(metrics)
keep <- !(metrics %in% c('LogLoss', 'LogLoss', 'AUC', 'Weighted LogLoss'))
length(x)
x <- x[keep]
length(x)
metrics <- unlist(sapply(x, '[[', 'metric'))
table(metrics)

#Remove weights/offset/exposure, Change to MAE
x <- lapply(x, function(a){
  a$weights <- NULL
  a$metric <- 'MAE'
  a
})

#Save
x <- as.yaml(x)
cat(x, file='~/workspace/mbtest-datasets/mbtest_datasets/data/Functionality/mae_with_preds.yaml')
