library(data.table)
library(anytime)
fast_date <- function(x, fmt){
  x_unique <- sort(unique(x))
  x_map <- match(x, x_unique)
  x_date <- as.Date(x_unique, format=fmt)
  x_date[x_map]
}
count_gaps <- function(url, fmt='%Y-%m-%d'){
  dat <- fread(url)
  dat_agg <- dat[,list(date=sort(unique(date))), by='id']
  dat_agg[,date:=fast_date(date, fmt=fmt)]
  setkeyv(dat_agg, c('id', 'date'))
  gaps <- dat_agg[,list('gap'=diff(date)), by='id']
  gaps <- gaps[,list(count = .N), by=c('id', 'gap')]
  gaps[,pct := count/sum(count), by='id']
  
  
  gap_count <- dat_agg[,list(diff=sort(unique(diff(date)))), by='id']
  out <- gap_count[,list(.N),by='id'][,sum(N>2)/sum(.N)]
  paste0('PCT of datasets with >2 gaps: ', round(out*100, 1), '%')
}
count_gaps('https://s3.amazonaws.com/datarobot_public_datasets/ny_toll_vehicle_count_by_day_2006-2013.csv', fmt='%m/%d/%Y')
count_gaps('https://s3.amazonaws.com/datarobot_public_datasets/Sample_Superstore.csv', fmt='%m/%d/%y')
count_gaps('https://s3.amazonaws.com/datarobot_public_datasets/SP500_multi.csv')
#count_gaps('https://s3.amazonaws.com/datarobot_public_datasets/Restaurant_Sales_in_Colorado-dr.csv')
count_gaps('https://s3.amazonaws.com/datarobot_public_datasets/walmart_just_sales.csv', fmt='%m/%d/%y')
#count_gaps('https://s3.amazonaws.com/datarobot_public_datasets/ny_toll_vehicle_count_by_hour_2013.csv')
count_gaps('https://s3.amazonaws.com/datarobot_public_datasets/rossman_store_sales.csv', fmt='%m/%d/%y')
count_gaps('https://s3.amazonaws.com/datarobot_public_datasets/stocks_clean_panel.csv', fmt='%m/%d/%y')
count_gaps('https://s3.amazonaws.com/datarobot_public_datasets/SP500_with_technical_multi.csv', fmt='%Y-%m-%d')
