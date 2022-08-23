#https://www.iso-ne.com/isoexpress/web/reports/load-and-demand/-/reports/dmnd-rt-hourly-sys?p_auth=6zuc30WN
right_now <- format(Sys.Date(), '%Y%m%d')
iso_ne_raw <- fread(paste0('https://www.iso-ne.com/transform/csv/hourlysystemdemand?start=20080701&end=', right_now), fill=T, skip=5, drop='H')

iso_ne <- copy(iso_ne_raw)
iso_ne <- iso_ne[!is.na(MWh),]
iso_ne[,Date := as.Date(Date, format='%m/%d/%Y')]
iso_ne[,Date := as.character(Date)]
iso_ne <- iso_ne[,list(date=Date, HE, y=MWh)]
write_csv(iso_ne, paste0('~/datasets/time_series/iso_ne', right_now, '.csv'))