library(sas7bdat)
library(data.table)
dat = sas7bdat::read.sas7bdat('~/Downloads/airlines_10gb_in_mem_5gb.sas7bdat')
dat = data.table(dat)
