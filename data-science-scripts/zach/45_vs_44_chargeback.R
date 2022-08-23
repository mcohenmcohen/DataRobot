library(data.table)

d45_raw <- fread('http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests=5bf32e247347c9002abad50a&max_sample_size_only=false')

d44_raw <- fread('http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests=5b88417d7347c9002c1da5ab&max_sample_size_only=false')

d45 <- d45_raw[Filename == 'chargeback_clean_80.csv',]
d44 <- d44_raw[Filename == 'chargeback_clean_80.csv',]

keys <- c('main_task')
# keys <-  c('main_task', 'Sample_Pct')
d45_agg <- d45[,list(.N), by=keys]
d44_agg <- d44[,list(.N), by=keys]
joined <- merge(d45_agg, d44_agg, by=keys, all=T, suffixes=c('45', '44'))
joined[,diff := N45 - N44]
joined[order(diff, N45, N44),]
