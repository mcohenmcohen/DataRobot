# Setup
rm(list=ls(all=T))
gc(reset=T)
library(data.table)
library(anytime)

# Load Data
data <- fread('~/workspace/data-science-scripts/zach/Predicted vs. Actual RAM - prod.csv')
setnames(data, gsub('@', '', names(data), fixed=T))
setnames(data, gsub('fields.', '', names(data), fixed=T))

data[,timestamp := gsub('st', '', timestamp)]
data[,timestamp := gsub('nd', '', timestamp)]
data[,timestamp := gsub('rd', '', timestamp)]
data[,timestamp := gsub('th', '', timestamp)]
data[,timestamp := tolower(timestamp)]
ts = as.POSIXct(strptime(data[['timestamp']], '%B %d %Y, %H:%M:%S'))
data[,timestamp := ts]

for(v in c('predicted_ram', 'actual_ram')){
  x <- data[[v]]
  x <- as.numeric(gsub(",", "", x, fixed = T))
  set(data, j=v, value=x)
}

# Analysis inputs
prediction_coefficient <- 3
prediction_constant <- 0
container_sizes <- c(
  61440,
  30720,
  20480,
  10240,
  5120)
container_sizes <- sort(container_sizes)

# Calculate RAM used under new logic
data[,buffered_ram := predicted_ram * prediction_coefficient + prediction_constant]
data[,new_container_id := findInterval(buffered_ram, c(0, container_sizes), all.inside = T)]
data[,table(new_container_id)]
data[,new_ram := container_sizes[new_container_id]]

# If we OOM, assume we pay the new cost while OOMing
# And then pay the old cost after we OOM and jump to 60GM
setorderv(data, 'timestamp')
data[new_ram < actual_ram & timestamp > as.POSIXct('2019-11-13'), ]
data[,sum(new_ram < actual_ram)]
data[,.N]
plot(new_ram < actual_ram ~ timestamp, data)
