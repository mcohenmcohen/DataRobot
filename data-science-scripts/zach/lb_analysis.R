stop()
rm(list=ls(all=T))
gc(reset=T)

library(data.table)
library(readr)
library(jsonlite)
library(pbapply)
library(compiler)

json_path <- '~/Downloads/4_month_lb_export.json'
#system(paste('head', json_path))

parse_line <- function(x){
  x <- tryCatch({
    x <- fromJSON(x, simplifyVector=F)
  }, error=function(e){
    warning(e)
    return(NULL)
  })
  out <- data.table(
    pid = unlist(x[['pid']]),
    lid = unlist(x[['_id']]),
    samplesize = x[['samplesize']],
    RMSE = unlist(x[['test']][['RMSE']]),
    blueprint = as.character(toJSON(x[['blueprint']]))
  )
  return(out)
}
parse_line <- cmpfun(parse_line)
str(parse_line(read_lines(json_path, n_max=1)))

N_RECORDS <- 35414597  # The mongo export tool told me this
CHUNK_SIZE <- 1e6
ITER_MAX <- floor(N_RECORDS/CHUNK_SIZE)
#ITER_MAX <- 3

data_list <- pblapply(0:ITER_MAX, function(i){
  raw_lines <- read_lines(json_path, skip=CHUNK_SIZE*i, n_max=CHUNK_SIZE)
  out <- pblapply(raw_lines, parse_line)
  out <- rbindlist(out, fill=T, use.names=T)
  return(out)
})

# Save data
file_path <- '~/Downloads/4_month_lb_export_parsed.csv'
data <- rbindlist(data_list, fill=T, use.names=T)
fwrite(data, file_path)

# Reload data
data <- fread(file_path)

# Analyze data
data[,project_id := as.integer(factor(pid, exclude=NULL))]
data[,leaderboard_id := as.integer(factor(lid, exclude=NULL))]
data[,blueprint_id := as.integer(factor(blueprint, exclude=NULL))]

keys <- c('blueprint_id', 'blueprint')
setkeyv(data, keys)
blueprint_map <- data[,list(.N), by=keys]


# 1e6 - 8 hours
# 1e5 - 
# 1e4 - 11 hours
# 1e3
# 1e2
# 1e1 - too slow