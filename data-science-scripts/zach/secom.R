library(data.table)
library(ggplot2)
library(ggthemes)
library(anytime)

dat_raw <- fread('https://s3.amazonaws.com/datarobot_data_science/test_data/secom_train80.csv')

dat <- copy(dat_raw)
dat[,V591 := anytime(V591)]
dat[,table(Target)]

ggplot(dat, aes(x=V591, y=Target)) + geom_smooth() + theme_tufte() + geom_point()
