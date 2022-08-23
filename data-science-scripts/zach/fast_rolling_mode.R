# Setup the problem
rm(list=ls(all=T))
gc(reset=T)
library(data.table)
set.seed(42)
N_ROWS = 4000000
N_SERIES = 10000
N_UNIQUE_VALUES = 5
ROLLING_WINDOW = 10

rows_per_series = as.integer(round(N_ROWS / N_SERIES))
data = CJ(series_id = 1:N_SERIES, time = 1:rows_per_series)
stopifnot(nrow(data) == N_ROWS)
data[,value := sample(1:N_UNIQUE_VALUES, .N, replace=T)]

# Solve the problem
start_time = Sys.time()

lookback_list = lapply(0:(ROLLING_WINDOW-1), function(lookback){
  data[,list(series_id, time=time+lookback, lookback=lookback, value=value)]
})
lookback_data = rbindlist(lookback_list)

setkeyv(lookback_data, c('series_id', 'time', 'value'))
lookback_counts = lookback_data[,list(count = .N), by=key(lookback_data)]

lookback_counts[,count := 0 - count]
lookback_counts[,tiebreak := runif(.N)]
setorderv(lookback_counts, c('series_id', 'time', 'count', 'tiebreak'))

lookback_counts[,id := 1:.N, by=c('series_id', 'time')]
rolling_modes = lookback_counts[id==1,]
rolling_modes = rolling_modes[,list(series_id, time, mode=value)]

print(rolling_modes)

end_time = Sys.time()
print(end_time - start_time)

# Check your work
check_results = function(id=1, t=200){
  input_data = data[series_id==id,][(t-ROLLING_WINDOW):t,]
  output_data = rolling_modes[series_id==id,][t,]
  print(input_data)
  print(output_data)
}
check_results(1, 10)
check_results(100, 234)
