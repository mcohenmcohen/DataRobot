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
