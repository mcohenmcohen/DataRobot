library(data.table)
library(stringdist)
x = fread('~/datasets/fred_weekly_post_1965.csv')
target_idx = amatch(tolower('W.WGS1YR_1Year Treasury Constant Maturity Rate_csv'), tolower(names(x)), method='cosine')
date_idx = amatch(tolower('date'), tolower(names(x)), method='cosine')
setnames(x, paste0('V', 1:ncol(x)))
setnames(x, names(x)[target_idx], 'target')
setnames(x, names(x)[date_idx], 'date')
first <- c('target', 'date')
setcolorder(x, c(first, setdiff(names(x), first)))
x[,sum(is.na(target)) / .N]
x <- x[!is.na(target),]
fwrite(x,'~/datasets/fred_weekly_post_1965_clean.csv' )