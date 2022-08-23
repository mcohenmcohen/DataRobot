library(data.table)
fileName <- '~/workspace/data-science-scripts/zach/jobs.txt'
x <- readChar(fileName, file.info(fileName)$size)
x <- strsplit(x, ',')[[1]]
x <- strsplit(x, ':')
x <- rbindlist(lapply(x, function(x){
  data.table(
    blueprint_id = x[1],
    dataset_id = x[2],
    sample_size = x[3],
    partition = x[4],
    k = x[5]
  )
}))
x[,run := as.integer(duplicated(paste(blueprint_id, sample_size)))]
x[44:84, run := 1L]

x[,list(N=.N), by=c('sample_size', 'run')]
