#Graffiti data
library(data.table)
library(bit64)
set.seed(1234)
dat <- fread('~/datasets/BostonGraffitiRemovalInOneDayCleaned.csv')

dat[closedinoneday == 0, closedinoneday := -1]

dat[,vw := paste0(
  closedinoneday,
  ' |t ', gsub(':', '', case_title),
  ' |l ', location,
  ' |ln', location_street_name,
  ' |s', source,
  ' |n ', neighborhood,
  ' |f ', fire_district,
  ' pd_', pwd_district,
  ' ccd_', city_council_district,
  ' pd_', police_district,
  ' nsd_', neighborhood_services_district,
  ' ', gsub(' ', '_', ward),
  ' p_', precinct,
  ' lu_', land_usage,
  ' zip_', location_zipcode,
  ' pt_', property_type,
  ' pid_', property_id,
  ' lat:', latitude,
  ' lon:', longitude
)]
#cat(head(dat$vw))

all_rows <- 1:nrow(dat)
trainrows <- sample(all_rows, .75*nrow(dat))
testrows <- setdiff(all_rows, trainrows)

train <- dat[trainrows,]
test <- dat[testrows,]

cat(train$vw, file='~/datasets/vw_graffiti_train.txt', sep='\n')
cat(test$vw, file='~/datasets/vw_graffiti_test.txt', sep='\n')

#Diamonds data
data(diamonds)
dat <- data.table(diamonds)
dat[,vw := paste0(
  price,
  ' |f',
  ' carat:', carat,
  ' cut', cut,
  ' color', color,
  ' clarity', clarity,
  ' depth:', depth,
  ' table:', table,
  ' price:', price,
  ' x:', x,
  ' y:', y,
  ' z:', z
)]

all_rows <- 1:nrow(dat)
trainrows <- sample(all_rows, .75*nrow(dat))
testrows <- setdiff(all_rows, trainrows)

trainrows <- sample(trainrows, 2000)
testrows <- sample(testrows, 2000)

train <- dat[trainrows,]
test <- dat[testrows,]

cat(train$vw, file='~/datasets/vw_diamonds_train.txt', sep='\n')
cat(test$vw, file='~/datasets/vw_diamonds_test.txt', sep='\n')

#VW models
#Quantile autolink
unlink('~/datasets/vw.cache')
system('vw --loss_function quantile --autolink 3 --random_seed 1234 --learning_rate 10.0 --decay_learning_rate 0.99 --quantile_tau 0.5 --bit_precision 18 --power_t 0.5 --l2 1e-10 --l1 0.01 --passes 100 --final_regressor ~/datasets/vwmodel.vw --cache_file ~/datasets/vw.cache -d ~/datasets/vw_diamonds_train.txt')
system('vw --testonly --initial_regressor ~/datasets/vwmodel.vw --predictions ~/datasets/vwpred.txt -d ~/datasets/vw_diamonds_test.txt --link identity')
p <- fread('~/datasets/vwpred.txt')
summary(p)

#lrq
unlink('~/datasets/vw.cache')
system('vw --random_seed 1234 --learning_rate .1 --decay_learning_rate 0.99 --bit_precision 18 --power_t 0.5 --l2 1e-10 --l1 0.01 --passes 10 --final_regressor ~/datasets/vwmodel.vw --cache_file ~/datasets/vw.cache -d ~/datasets/vw_diamonds_train.txt --lrq ::2 --bfgs')
system('vw --testonly --initial_regressor ~/datasets/vwmodel.vw --predictions ~/datasets/vwpred.txt -d ~/datasets/vw_diamonds_test.txt --link identity')
p <- fread('~/datasets/vwpred.txt')
summary(p)

#Quantile autolink
unlink('~/datasets/vw.cache')
system('vw --loss_function quantile --autolink 3 --random_seed 1234 --learning_rate 10.0 --decay_learning_rate 0.99 --quantile_tau 0.5 --bit_precision 18 --power_t 0.5 --l2 1e-10 --l1 0.01 --passes 100 --final_regressor ~/datasets/vwmodel.vw --cache_file ~/datasets/vw.cache -d ~/datasets/vw_diamonds_train.txt')
system('vw --testonly --initial_regressor ~/datasets/vwmodel.vw --predictions ~/datasets/vwpred.txt -d ~/datasets/vw_diamonds_test.txt --link identity')
p <- fread('~/datasets/vwpred.txt')
summary(p)
