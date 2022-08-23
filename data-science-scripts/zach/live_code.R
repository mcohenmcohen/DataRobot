##################################################################
# Setup
##################################################################

#Install all the packages at once
install.packages(c('ranger', 'ggplot2', 'data.table'), repos="http://cloud.r-project.org/")

#Install all the packages individually
install.packages('ranger')
install.packages('data.table')
install.packages('ggplot2')

#Notes:
# command + enter to run a block of code (option + enter on windows)
# control + L to clear the screen

#Make plots pretty
options(warn=-1)
library(scales)
my_alpha <- 0.4
point_color <- '#756bb1'
point_color <- 'blue'
color_scale <- c(
  "#1f78b4", "#ff7f00", "#6a3d9a", "#33a02c", "#e31a1c", "#b15928",
  "#a6cee3", "#fdbf6f", "#cab2d6", "#b2df8a", "#fb9a99", "#ffff99"
  )
color_scale <- c(
  "blue", "orange", "purple", "green", "red", "black", "yellow", "grey"
)

##################################################################
# ggplot2
##################################################################

#Simple diamonds examples
library(ggplot2)
data("diamonds")
summary(diamonds)
set.seed(42)
diamonds <- diamonds[sample(1:nrow(diamonds), 5000),]
head(diamonds)
summary(diamonds)
hist(diamonds$carat)
hist(log(diamonds$carat))
ggplot(diamonds, aes(x=carat, y=price)) +
  geom_point(alpha=my_alpha, color=point_color) +
  theme_bw() +
  scale_x_log10() +
  scale_y_log10(labels=comma) +
  geom_smooth() +
  geom_smooth(method='lm')

#Facets
ggplot(diamonds, aes(x=carat, y=price)) +
  geom_point(alpha=my_alpha, color=point_color) +
  theme_bw() +
  scale_x_log10() +
  scale_y_log10(labels=comma) +
  facet_grid(cut ~ color) +
  geom_smooth(method='lm')

#Hex bins
ggplot(diamonds, aes(x=carat, y=price)) + geom_hex() + theme_bw()

#Boxplot
ggplot(diamonds, aes(x=color, y=price)) + geom_jitter()
ggplot(diamonds, aes(x=color, y=price)) + geom_boxplot() +
  theme_bw() +
  scale_y_log10(labels=comma)

#Violinplot
ggplot(diamonds, aes(x=color, y=carat, fill=color)) + geom_violin() +
  theme_bw() + scale_y_log10() +
  scale_fill_manual(values = color_scale)

##################################################################
# data.table
##################################################################
# Alternative: dplyr
library(data.table)
dat <- data.table(diamonds)
dat[carat==1, mean(price), by='cut']
dat[carat==1, list(
  price = mean(price),
  count = .N
), by='cut'][order(cut),]

dat[carat==1,list(
  price = round(median(price)),
  count = .N
), by='cut'][order(cut),]

dat[,cor(price, carat), by='cut'][order(cut)]
dat[,as.list(coef(lm(price ~ carat + x + y + z))), by='cut']
dat[,as.list(coef(lm(price ~ carat + x + y + z))), by='cut']

#Some excercises:
# mean/median/sd of price by group
# aggregate by 2 groups
# Find the most expensive diamond
dat[,list(
  mean = mean(price),
  med = median(price),
  sd = sd(price)
), by='cut'][order(cut),]

dat[,list(mean = mean(price)), by=c('cut', 'color')][order(mean),]
dat[,list(
  price = mean(price),
  carat = mean(carat)
  ), by=c('cut', 'color')][order(price),]

dat[price == max(price),]

##################################################################
# A real example
##################################################################

