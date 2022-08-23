library(jsonlite)
library(data.table)
library(httr)

score_file <- function(
  file_name = 'tempchunkfile1.csv',
  model = '55d5de76e2d7ab0111d702e8',
  project = '55d5dda275f6c03b8e66a27d',
  token = '5KV04p-zR9Jx2GfP36hirPxYBjZEsvuQ',
  base_url = 'https://app.datarobot.com/api/v2/projects'
){

  #Check params
  if(model == project){
    stop("Same hash supplied for model and project, one is probably wrong.")
  }

  #Construct parameters
  url <- paste(base_url, project, 'models', model, 'predict', sep='/')
  url <- paste0(url, "/")
  dat <- toJSON(fread(file_name))
  token <- paste('Token', token)

  #Make POST
  res <- POST(
    url,
    add_headers(Authorization = 'Token 5KV04p-zR9Jx2GfP36hirPxYBjZEsvuQ'),
    body = dat,
    content_type_json())
  stop_for_status(res) #Fail on status other than 200

  #Extract content
  return(fromJSON(content(res, as='text')))
}

p <- score_file('tempchunkfile1.csv')
p[1:3]
head(p$predictions)
