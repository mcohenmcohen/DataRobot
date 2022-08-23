library(data.table)
dat <- fread('http://shrink.drdev.io/api/leaderboard_export/advanced_export.csv?mbtests=5f1a1ee1add5affe6b2e2af0&max_sample_size_only=false')

colnames <- sort(names(dat))
print(colnames[grep('vertex', tolower(colnames))])

dat[,storage_size_mb := blueprint_storage_size_P1 / 1000000]
summary(dat[,storage_size_mb])
dat[!is.na(storage_size_mb),list(
  min=min(storage_size_mb),
  pct_1=quantile(storage_size_mb, .05),
  mean=mean(storage_size_mb),
  median=median(storage_size_mb),
  pct_99=quantile(storage_size_mb, .99),
  max=max(storage_size_mb)
)]