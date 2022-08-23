library(data.table)
library(readr)
dat <- fread('https://s3-ap-southeast-1.amazonaws.com/datarobotfiles/DR_Demo_Statistical_Case_Estimates.csv')
dat[,LogInitialCaseEstimate := log(InitialCaseEstimate)]
dat[,LogWeeklyRate := log1p(WeeklyRate)]
dat[,LogHoursWorkedPerWeek := log1p(HoursWorkedPerWeek)]
dat[,InitialCaseEstimate := NULL]
dat[,WeeklyRate := NULL]
dat[,HoursWorkedPerWeek := NULL]
stopifnot(all(is.finite(dat[['LogInitialCaseEstimate']])))
stopifnot(all(is.finite(dat[['LogWeeklyRate']])))
stopifnot(all(is.finite(dat[['LogHoursWorkedPerWeek']])))
write_csv(dat, '~/datasets/DR_Demo_Statistical_Case_Estimates.csv')
