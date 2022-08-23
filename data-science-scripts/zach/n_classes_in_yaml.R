library(yaml)
library(data.table)
library(readr)
library(httr)
library(pbapply)
#https://stackoverflow.com/a/20921907

rm(list=ls(all=T))
gc(reset=T)
set.seed(42)

handle_nothing <- function(x) x
custom_handlers <- list(
  "bool#y"=handle_nothing,
  "bool#yes"=handle_nothing,
  "bool#T"=handle_nothing,
  "bool#TRUE"=handle_nothing,
  "bool#1"=handle_nothing
)

validate_yaml <- function(a){

  #Check for weird file types
  check_1 <- grepl('.xls', a$dataset_name, fixed=T)
  check_2 <- grepl('.zip', a$dataset_name, fixed=T)
  check_3 <- a$dataset_name %in% c(
    'https://s3.amazonaws.com/datarobot_public_datasets/stack_overflow_closed_question_1Gb.csv',
    'https://s3.amazonaws.com/datarobot_public_datasets/subreddit_15_classes.csv',
    'https://s3.amazonaws.com/datarobot_public_datasets/fakenewschallenge.csv'
  )

  #Determine the file prefix
  if(grepl('openML/large/', a$dataset_name, fixed=T)){
    prefix <- 'https://s3.amazonaws.com/datarobot_public_datasets/openML/large/'
  } else if(grepl('openML/datasets/', a$dataset_name, fixed=T)){
    prefix <- 'https://s3.amazonaws.com/datarobot_public_datasets/openML/datasets/'
  } else if(grepl('libsvm/datasets/', a$dataset_name, fixed=T)){
    prefix <- 'https://s3.amazonaws.com/datarobot_public_datasets/libsvm/datasets/'
  } else if(grepl('text/datasets/', a$dataset_name, fixed=T)){
    prefix <- 'https://s3.amazonaws.com/datarobot_public_datasets/text/datasets/'
  } else if(grepl('R/datasets/', a$dataset_name, fixed=T)){
    prefix <- 'https://s3.amazonaws.com/datarobot_public_datasets/R/datasets/'
  } else if(grepl('UCI/datasets/', a$dataset_name, fixed=T)){
    prefix <- 'https://s3.amazonaws.com/datarobot_public_datasets/UCI/datasets/'
  } else if(grepl('kaggle/', a$dataset_name, fixed=T)){
    prefix <- 'https://s3.amazonaws.com/datarobot_public_datasets/kaggle/'
  } else{
    prefix <- 'https://s3.amazonaws.com/datarobot_public_datasets/'
  }
  stopifnot(grepl(prefix, a$dataset_name, fixed=T))

  #URL encode non-prefix
  x <- gsub(prefix, 'datasets/', a$dataset_name, fixed=T)
  x <- gsub(prefix, 'large/', a$dataset_name, fixed=T)
  x <- gsub(prefix, 'datasets/', a$dataset_name, fixed=T)
  x <- gsub(prefix, '', a$dataset_name, fixed=T)
  x <- utils::URLencode(x, reserved=T)
  x <- paste0(prefix, x)
  #print(paste0('...', x))

  #If it's a weird file, just return
  if(check_1 | check_2 | check_3){
    return(NA)
  }
  Sys.sleep(1)
  dat <- fread(
    paste('curl -s', utils::URLencode(x)),
    header=T,
    strip.white=F,
    colClasses='string',
  )
  return(length(unique(dat[[a[['target']]]])))
}

# Load yamls
in_full <- yaml.load_file("~/workspace/mbtest-datasets/mbtest_datasets/data/Functionality/mbtest_data_multiclass.yaml", handlers=custom_handlers)
in_100 <- yaml.load_file("~/workspace/mbtest-datasets/mbtest_datasets/data/Functionality/mbtest_data_multiclass_upto_100mb.yaml", handlers=custom_handlers)

# Determine number of classe
classes <- pbsapply(in_full, validate_yaml)
stopifnot(length(classes) == length(in_full))
summary(classes)
hist(classes)

# Save class count
class_table <- data.table(
  dataset_name = sapply(in_full, '[[', 'dataset_name'),
  classes = classes
)
TABLE_FILE <- '~/workspace/data-science-scripts/zach/multiclas_yaml_class_count.csv'
write_csv(class_table, TABLE_FILE)
# class_table <- fread(TABLE_FILE)

# Save <= 12 classes
keep <- class_table[classes <= 10, dataset_name]
out_full <- in_full[sapply(in_full, '[[', 'dataset_name') %in% keep]
out_100 <- in_100[sapply(in_100, '[[', 'dataset_name') %in% keep]

out_full <- yaml::as.yaml(out_full)
out_100 <- yaml::as.yaml(out_100)

outfile <- "~/workspace/mbtest-datasets/mbtest_datasets/data/Functionality/mbtest_data_multiclass_up_to_10_classes.yaml"
cat("# Multiclass files that are <=10 classes\n", file=outfile, append=F)
cat(out_full, file=outfile, append=T)

outfile <- "~/workspace/mbtest-datasets/mbtest_datasets/data/Functionality/mbtest_data_multiclass_upto_100mb_up_to_10_classes.yaml"
cat("# Multiclass files that are <100MB and <=10 classes\n", file=outfile, append=F)
cat(out_100, file=outfile, append=T)
