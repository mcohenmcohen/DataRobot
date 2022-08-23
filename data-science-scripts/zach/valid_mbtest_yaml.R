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
  print(paste0('...', x))

  #Load the raw data, and check that it's not an access error
  Sys.sleep(2)
  raw <- system(paste('curl -s', x, '| head -10'), intern=T)
  access_check <- grepl('<Code>AccessDenied</Code>', paste(raw, collapse=' '), fixed=T)
  if(access_check){
    print(a$dataset_name)
    stop('Access Denined.  Please make file public or re-upload')
  }

  #Check the size of the file
  out <- HEAD(x)[['headers']][['content-length']]
  out <- as.numeric(out) / (1024 * 1024)

  #If it's a weird file, and we don't have an access error, just return
  if(check_1 | check_2 | check_3){
    if(length(raw) > 5){
      return(out)
    } else{
      print(a$dataset_name)
      print(raw)
      stop(paste(raw, collpase=T))
    }
  }
  Sys.sleep(1)
  dat <- fread(
    paste('curl -s', utils::URLencode(x), '| head -110'),
    header=T,
    strip.white=F,
    colClasses='string',
  )
  if(nrow(dat) < 100){
    print(x)
    warning(paste('Data Too Small:', x))
  }
  if(ncol(dat) > 20000){
    print(x)
    warning(paste('Data Has Too Many Columns:', x))
  }
  if(! a$target %in% names(dat)){
    print(a$dataset_name)
    print(paste('...', paste(names(dat), collapse=' ')))
    print(head(dat))
    stop(paste(
      "Target value",
      a$target,
      'not in dataset:',
      a$dataset_name
    ))
  }
  if(length(unique(dat[[a$target]])) < 3){
    #print(paste(x, 'Too few classes?'))
    #warning(x)
    #warning('Too few classes')
  }
  return(out)
}

N <- 0
validate_mbtest_yaml <- function(x){
  x_parsed <- yaml.load_file(x, handlers=custom_handlers)
  N <<- length(x_parsed) #LOL
  yaml_list <<- x_parsed #LOL
  sapply(x_parsed, validate_yaml)
}

fn <- "~/workspace/mbtest-datasets/mbtest_datasets/data/Functionality/mbtest_data_multiclass.yaml"
sizes <- validate_mbtest_yaml(fn)
stopifnot(all(is.finite(sizes)))
stopifnot(length(sizes) == N)
print(N)

#Save ones smaller than 100MB:
x_parsed <- yaml.load_file(fn, handlers=custom_handlers)
warnings()
#sapply(x_parsed, '[[', 'dataset_name')[sizes>=100]
summary(sizes)
hist(sizes)
sum(sizes < 100)

x_parsed_out <- x_parsed[order(sizes)]
sizes_out <- sizes[order(sizes)]

outfile <- "~/workspace/mbtest-datasets/mbtest_datasets/data/Functionality/mbtest_data_multiclass_upto_100mb.yaml"
x_parsed_out <- x_parsed_out[sizes < 100]
x_parsed_out <- yaml::as.yaml(x_parsed_out)
x_parsed_out <- gsub("'True'", "True", x_parsed_out, fixed=T)
cat("# Multiclass files that are <100MB, for testing with the old one-vs-all multiclass script\n", file=outfile, append=F)
cat(x_parsed_out, file=outfile, append=T)

length(x_parsed)
length(x_parsed[sizes < 100])
