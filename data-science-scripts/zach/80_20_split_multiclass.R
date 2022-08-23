library(yaml)
library(data.table)
library(readr)
library(httr)
library(pbapply)

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

  #If it's a weird file, return nothing.  Skip the 80/20 split
  if(check_1 | check_2 | check_3){
    return(NULL)
  }

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
  filename <- gsub(prefix, '', a$dataset_name, fixed=TRUE)
  x <- utils::URLencode(filename, reserved=T)
  x <- paste0(prefix, x)
  print(paste0('...', x))

  #Check if we already downloaded the data
  prefix <- '~/datasets/multiclass/'
  train_name <- gsub('.csv', '_80.csv', filename, fixed=T)
  test_name <- gsub('.csv', '_20.csv', filename, fixed=T)
  train_file <- paste0(prefix, train_name)
  test_file <- paste0(prefix, test_name)

  if(!(file.exists(train_file) & file.exists(test_file))){
    print('...Downloading and splitting')
    #Load data
    dat <- fread(
      x,
      header=T,
      strip.white=F,
      showProgress=F
    )

    #80/20 split
    set.seed(32860)
    dat <- dat[sample(1:.N),]
    split <- dat[,floor(.N *.80)]

    #Save
    write_csv(dat[1:split,], train_file)
    write_csv(dat[(split+1):.N,], test_file)
  } else{
    print('...Skipping Download')
  }

  #Construct and return new yaml
  prefix <- 'https://s3.amazonaws.com/datarobot_public_datasets/multiclass/'
  a[['dataset_name']] <- paste0(prefix, train_name)
  a[['prediction_dataset_name']] <- paste0(prefix, test_name)
  return(a)
}

filename <- "~/workspace/mbtest-datasets/mbtest_datasets/data/Functionality/mbtest_data_multiclass_up_to_10_classes.yaml"
yaml_in <- yaml.load_file(filename, handlers=custom_handlers)
yaml_out_full <- pbsapply(yaml_in, validate_yaml)
yaml_out <- yaml_out_full[!sapply(yaml_out_full, is.null)]

outfile <- "~/workspace/mbtest-datasets/mbtest_datasets/data/Functionality/mbtest_data_multiclass_up_to_10_classes_with_preds.yaml"
yaml_out <- yaml::as.yaml(yaml_out)
yaml_out <- gsub("'True'", "True", yaml_out, fixed=T)
cat("# Multiclass files with <= 10 classes + predictions\n", file=outfile, append=F)
cat(yaml_out, file=outfile, append=T)
