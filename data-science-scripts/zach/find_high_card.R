library(data.table)

########################################################################
# Recentish current with preds - zero inflaed class
########################################################################
library(data.table)
aaa=fread('http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests=5c41fbaf7347c9002bf34388&max_sample_size_only=false')
agg = aaa[Y_Type == 'Binary', list(y_zero=max(y_zero, na.rm=T), rows=max(rows, na.rm=T)), by='display_dataset_name']
agg[,pct_zero := y_zero / rows]
agg[order(-pct_zero),][pct_zero>.80, display_dataset_name]

########################################################################
# Recent current with preds - high card
########################################################################
library(data.table)

aaa=fread('http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests=5c41fbaf7347c9002bf34388&max_sample_size_only=false')
agg = aaa[,list(
  cat_card_max=max(cat_card_max),
  dataset_x_cat=max(dataset_x_cat),
  dataset_x_cols=max(dataset_x_cols)), by='Filename']

# Wide data
agg[order(dataset_x_cols, decreasing=TRUE)[1:10],list(Filename, dataset_x_cols)]

# High Card
agg[dataset_x_cat > 1 & !is.na(cat_card_max),][order(dataset_x_cols, decreasing=TRUE)[1:10],list(Filename, cat_card_max)]

# Wide data
agg[order(dataset_x_cols, decreasing=TRUE)[1:10],list(Filename, dataset_x_cols)]

########################################################################
# Enterprise tests - high card
########################################################################
# https://datarobot.atlassian.net/wiki/spaces/QA/pages/111691866/Release+MBTests

library(data.table)
library(pbapply)
mbtest_ids <- c(
  '5c40da04a2c902000142a185',
  '5c4243e5e3cb9e0001416923',
  '5c4b4ed8ba08880001b81d9b',
  '5c40e4b330cf35000175821c',
  '5c4243c42ce6130001323a60',
  '5c461f85970b8900011b19d4'

)
prefix = 'http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests='
suffix = '&max_sample_size_only=false'
mbtest_urls <- paste0(prefix, mbtest_ids, suffix)
dat_raw <- pblapply(mbtest_urls, fread)
dat <- rbindlist(lapply(dat_raw, function(x) x[,list(Filename, dataset_x_cols)]))
agg = dat[,list(dataset_x_cols=max(dataset_x_cols)), by='Filename']
agg[order(dataset_x_cols, decreasing=TRUE)[1:10],list(Filename, dataset_x_cols)]

