library(data.table)

########################################################################
# Recent current with preds
########################################################################
library(data.table)

aaa=fread('http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests=5c41fbaf7347c9002bf34388&max_sample_size_only=false')

aaa[,gini_v := as.numeric(`Gini Norm_P1`)]
aaa[,gini_h := as.numeric(`Gini Norm_H`)]
aaa[,gini_cv := as.numeric(`Gini Norm_P1-5`)]

########################################################################
# Tree models don't do well
########################################################################

agg <- copy(aaa)
agg[grepl('XGB', main_task), main_task := 'tree']
agg[grepl('LGBM', main_task), main_task := 'tree']
agg[grepl('GBM', main_task), main_task := 'tree']
agg[grepl('GBR', main_task), main_task := 'tree']
agg[grepl('GBC', main_task), main_task := 'tree']

agg = agg[as.numeric(Sample_Pct) <= 64,list(
  gini = max((gini_v + gini_h)/2),
  gini_cv = max(gini_cv))
  , by=c('Filename', 'main_task')]

agg[!is.na(gini_cv), gini := gini_cv]
agg <- agg[!is.na(gini),]
agg[,gini_cv := NULL]

agg[,max_gini := max(gini), by='Filename']
agg[,diff := max_gini - gini]
setorder(agg, diff)
agg[main_task == 'tree',]

########################################################################
# Lots of text
########################################################################

agg = aaa[,list(
  cat_card_max=max(cat_card_max),
  dataset_x_cat=max(dataset_x_cat),
  dataset_x_cols=max(dataset_x_cols),
  dataset_x_txt=max(dataset_x_txt))
  , by='Filename']

# Wide data
agg[order(dataset_x_txt, decreasing=TRUE)[1:10],list(Filename, dataset_x_txt)]

