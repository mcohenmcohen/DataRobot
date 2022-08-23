#!/usr/bin/env Rscript
args <- commandArgs(TRUE)
stopifnot(length(args) == 1)
stopifnot(is.character(args[[1]]))

library(data.table)
library(readr)
filename <- args[[1]]
dat <- fread(filename, nrows=2)
oldnames <- names(dat)
newnames <- make.names(oldnames, unique = TRUE, allow_ = FALSE)
changed <- oldnames != newnames
if(any(changed)){
  for(i in which(changed)){
    print(paste0("Change column '", oldnames[i], "' to '", newnames[i], "'"))
  }
}

httr::GET("http://httpbin.org/delay/3", httr::timeout(1))
