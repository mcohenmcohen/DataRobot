rm(list=ls(all=T))
gc(reset=T)

library(data.table)
library(ggplot2)
library(readr)

x <- fread('curl -s https://s3.amazonaws.com/datarobot_data_science/text_data/amazon_review_polarity_full.csv | head -10001')
x <- x[order(nchar(text), nchar(title)),]
x[,date := seq.POSIXt(as.POSIXct('2010-01-01'), by='hours', length.out=.N)]
x[,class_id := as.numeric(class_id)]
ggplot(x, aes(x=date, y=class_id)) + geom_smooth()

N1 <- x[,floor(.8*.N)]
N2 <- N1 + 1
train <- x[1:N1,]
test <- x[N2:.N,]

test[,class_id := as.character(class_id)]
test[,class_id := '']

write_csv(train, '~/datasets/amazon_review_polarity_full_time_series_text_train.csv')
write_csv(test,  '~/datasets/amazon_review_polarity_full_time_series_text_test.csv')

# https://s3.amazonaws.com/datarobot_public_datasets/amazon_review_polarity_full_time_series_text_train.csv
# https://s3.amazonaws.com/datarobot_public_datasets/amazon_review_polarity_full_time_series_text_test.csv
