
############################################################
# Libraries
############################################################
stop()
rm(list=ls(all=T))
gc(collect=T)
library(pbapply)
library(data.table)
library(memoise)

############################################################
# Download data
############################################################
mbtest_ids <- c(
  '60aae6e2e5a56a4e1a6f185b', # Core M
  '60b5c074b37e3e097fa88280', # Core DL
  '60cd962ef31021253ba8ea28'  # TS
)

download_data <- memoise(function(id){
  prefix <- 'http://shrink.drdev.io/api/leaderboard_export/advanced_export.csv?mbtests='
  suffix <- '&max_sample_size_only=false'
  out <- fread(paste0(prefix, id, suffix))
  set(out, j='mbtestid', value=id)
}, cache = cachem::cache_disk("cache/"))

dat_raw <- pblapply(mbtest_ids, download_data)

############################################################
# Analyze data
############################################################
# See also https://github.com/datarobot/mbtest_white_paper/blob/main/README.Rmd

# Join the test results together
data <- rbindlist(dat_raw)

# Clean the recommendation_types column
data[,rec_for_deploy := as.numeric(grepl('RECOMMENDEDFORDEPLOYMENT', recommendation_types, fixed=T))]

# Drop projects with no recommended model
good_pids <- data[,list(N=sum(rec_for_deploy)), by=pid][N > 0,sort(unique(pid))]
data <- data[pid %in% good_pids,]

# Keep unique files
unique_files <- data[,list(.N), by=c('Filename', 'pid')]
setorder(unique_files, Filename, -N, pid)
unique_pids <- unique_files[!duplicated(Filename), pid]
data <- data[pid %in% unique_pids,]

# Subset to reccomended model only
data <- data[rec_for_deploy == 1,]

# Summarize the metrics
# sort(names(data))
#x = names(data); sort(x[grepl('_H$', x)])

# Combine weighted/unweighted metrics
metrics <- c('R Squared_H', 'FVE Binomial_H', 'FVE Poisson_H', 'FVE Tweedie_H', 'FVE Gamma_H')
for(m in metrics){
  reg <- as.numeric(data[[m]])
  weighted <- as.numeric(data[[paste('Weighted', m)]])
  idx <- is.na(reg)
  reg[idx] <- weighted[idx]
  set(data, j=m, value=reg)
}

# Subset to only the columns we want
data <- data[,c('pid', 'metric', 'Y_Type', metrics), with=F]
data[,metric := gsub('Weighted ', '', metric)]

# subset to only binary/reg for now
data <- data[! Y_Type %in% c('Multiclass', 'Multilabel'),]  # Shrink doesn't have these FVE metrics yet
data[,table(metric, Y_Type)]

# Remove one bad row
data <- data[`FVE Binomial_H` > 784746166277798 | is.na(`FVE Binomial_H`),]

# Choose FVE metric based on project metrics
data[,normalized_accuracy := `R Squared_H`]
data[Y_Type == 'Binary', normalized_accuracy := `FVE Binomial_H`]
data[Y_Type == 'Regression' & metric == 'Poisson Deviance', normalized_accuracy := `FVE Poisson_H`]
data[Y_Type == 'Regression' & metric == 'Tweedie Deviance', normalized_accuracy := `FVE Tweedie_H`]
data[Y_Type == 'Regression' & metric == 'Gamma Deviance', normalized_accuracy := `FVE Gamma_H`]

# Remove unknowns
data <- data[!is.na(normalized_accuracy),]

# Bucket the accuracy
data[,normalized_accuracy_bucket := cut(normalized_accuracy, breaks=c(-5, 0, seq(.1, 1, by=.1)))]
data[,table(normalized_accuracy_bucket)]

# And now show the data
hist(data[['normalized_accuracy']], breaks=100, main='Normalized accuracy', xlab='Normalized accuracy')
data[,round(summary(normalized_accuracy), 2)]
