
############################################
# Graffiti
############################################

library(data.table)
library(bit64)
dat <- fread('~/datasets/BostonGraffitiRemovalInOneDay.csv')

dat[,fire_district := paste0('f', fire_district)]
dat[,city_council_district := paste0('c', city_council_district)]
dat[,neighborhood_services_district := paste0('n', neighborhood_services_district)]
dat[,precinct := paste0('p', precinct)]
dat[,LOCATION_ZIPCODE := paste0('z', LOCATION_ZIPCODE)]
dat[,Property_ID := paste0('id', Property_ID)]
dat[,Geocoded_Location := NULL]
dat[,CASE_STATUS := NULL]
dat[,CLOSURE_REASON := NULL]
setnames(dat, tolower(names(dat)))
write.csv(dat, '~/datasets/BostonGraffitiRemovalInOneDayCleaned.csv', row.names=FALSE)

############################################
# Iraq
############################################

library(data.table)
library(bit64)
dat <- fread('~/datasets/sentiment140_twitter_1txt_100k_test20.csv')
head(dat)

dat[,fire_district := paste0('f', fire_district)]
dat[,city_council_district := paste0('c', city_council_district)]
dat[,neighborhood_services_district := paste0('n', neighborhood_services_district)]
dat[,precinct := paste0('p', precinct)]
dat[,LOCATION_ZIPCODE := paste0('z', LOCATION_ZIPCODE)]
dat[,LOCATION_ZIPCODE := paste0('z', LOCATION_ZIPCODE)]
dat[,Property_ID := paste0('id', Property_ID)]
dat[,Geocoded_Location := NULL]
dat[,CASE_STATUS := NULL]
setnames(dat, tolower(names(dat)))
write.csv(dat, '~/datasets/BostonGraffitiRemovalInOneDayCleaned.csv', row.names=FALSE)



