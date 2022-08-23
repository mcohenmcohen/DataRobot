library(forecast)
library(Mcomp)
library(yaml)
library(pbapply)
library(readr)

M1
M3
#M3Forecast

summary(sapply(M1, function(x) length(x$x) + length(x$xx)))
summary(sapply(M3, function(x) length(x$x) + length(x$xx)))

sum(sapply(M1, function(x) length(x$x) + length(x$xx)) >= 140)
sum(sapply(M3, function(x) length(x$x) + length(x$xx)) >= 140)

S3_DIR <- 'https://s3.amazonaws.com/datarobot_public_datasets/time_series/'
mcomp_to_yaml <- function(m){
  out <- pblapply(m, function(x){
    ts <- c(x$x, x$xx)
    if(length(ts) > 140){
      ds_name <- make.names(x$description)
      out <- list(
        dataset_name = paste0(S3_DIR, ds_name, '.csv'),
        target = 'y',
        metric = 'RMSE',
        partitioning = list(
          partition_column = 'date',
          autopilot_data_selection_method = 'rowCount'
        )
      )
      return(out)
    }
  })
  out <- out[!is.null(out)]

}
