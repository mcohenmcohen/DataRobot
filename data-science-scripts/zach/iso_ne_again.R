library(data.table)
FILENAME <- "~/datasets/time_series/iso_ne_hourly_load.csv"
DATEVAR <- "date"
dat <- fread(FILENAME)
setkeyv(dat, DATEVAR)
dat <- dat[,DUPLICATE_DATE_ROW_ID := 1:.N, by=DATEVAR]
dat <- dat[DUPLICATE_DATE_ROW_ID==1,]
dat[, DUPLICATE_DATE_ROW_ID := NULL]
fwrite(dat, paste0(FILENAME, 'deduped.csv'))