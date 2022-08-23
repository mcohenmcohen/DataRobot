library(jsonlite)
library(data.table)
library(RCurl)

getURIs =
  function(uris, ..., multiHandle = getCurlMultiHandle(), .perform = TRUE)
  {
    content = list()
    curls = list()

    for(i in uris) {
      curl = getCurlHandle()
      content[[i]] = basicTextGatherer()
      opts = curlOptions(URL = i, writefunction = content[[i]]$update, ...)
      curlSetOpt(.opts = opts, curl = curl)
      multiHandle = push(multiHandle, curl)
    }

    if(.perform) {
      complete(multiHandle)
      lapply(content, function(x) x$value())
    } else {
      return(list(multiHandle = multiHandle, content = content))
    }
  }


score_files <- function(
  file_names = rep('tempchunkfile1.csv', 10),
  model = '55d5de76e2d7ab0111d702e8',
  project = '55d5dda275f6c03b8e66a27d',
  token = '5KV04p-zR9Jx2GfP36hirPxYBjZEsvuQ',
  base_url = 'https://app.datarobot.com/api/v2/projects',
  multiHandle = getCurlMultiHandle()
){

  #Check params
  if(model == project){
    stop("Same hash supplied for model and project, one is probably wrong.")
  }

  #Construct parameters
  url <- paste(base_url, project, 'models', model, 'predict', sep='/')
  url <- paste0(url, "/")

  #Built asyncronous handle pool
  content = list()
  curls = list()
  for(i in seq_along(file_names)) {
    curl = getCurlHandle()
    content[[i]] = basicTextGatherer()
    opts = curlOptions(
      URL = url,
      writefunction = content[[i]]$update
      )
    curlSetOpt(
      postfields = list(
        body = toJSON(fread(file_names[i])),
        Authorization = paste('Token', token)
      ),
      =
      .opts = opts,
      curl = curl)

    multiHandle = push(multiHandle, curl)
  }
  complete(multiHandle)

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

files <- rep('tempchunkfile1.csv', 10)
p <- score_file('tempchunkfile1.csv')
lapply(p, function(x) head(x$prediction))
