#Download course data: https://goo.gl/ejxsWF
#Shameless plug for my datacamp class on caret: https://www.datacamp.com/courses/machine-learning-toolbox

#Setup
library(scales)
my_alpha <- 0.4
point_color <- "#756bb1"
color_scale <- c("blue", "orange", "purple", "green", "red", "black", "grey", "yellow", "brown")
color_scale <-  c(
  "#1f78b4", "#ff7f00", "#6a3d9a", "#33a02c", "#e31a1c", "#b15928",
  "#a6cee3", "#fdbf6f", "#cab2d6", "#b2df8a", "#fb9a99", "#ffff99"
  )

#ggplot2
library(ggplot2)
data("diamonds")
head(diamonds)
tail(diamonds)
set.seed(42)
point_color <- "#756bb1"
diamonds <- diamonds[sample(1:nrow(diamonds), 5000),]

#Scatterplots
ggplot(diamonds, aes(x=carat, y=price)) +
  geom_point(alpha=my_alpha, color = point_color) +
  theme_bw() +
  scale_x_log10() +
  scale_y_log10() +
  # geom_smooth() +
  geom_smooth(method='lm', color='black') +
  facet_grid(cut ~ clarity)
  # facet_wrap( ~ cut)

#Boxplots
ggplot(diamonds, aes(x=color, y=price)) + geom_point()
ggplot(diamonds, aes(x=color, y=price)) + geom_jitter()
ggplot(diamonds, aes(x=color, y=price)) + geom_boxplot() + scale_y_log10() + theme_bw()
ggplot(diamonds, aes(x=color, y=carat)) + geom_boxplot() + scale_y_log10() + theme_bw()
ggplot(diamonds, aes(x=color, y=price)) + geom_violin() + theme_bw() + scale_y_log10()

ggplot(diamonds, aes(x=color, y=price, fill=color)) +
  geom_violin(draw_quantiles=c(.80, .90, .99)) + theme_bw() + scale_y_log10() +
  scale_fill_manual(values=color_scale) +
  facet_wrap( ~ cut)

ggplot(diamonds, aes(x=color, y=cut)) + theme_bw() +
  stat_bin2d() + ggtitle('This is my plot') +

ggplot(diamonds, aes(x=color, y=cut, fill=carat)) + theme_bw() + geom_tile()

ggplot(diamonds, aes(x=carat, y=price)) + theme_bw() + stat_bin2d() +
  scale_x_log10() + scale_y_log10()

ggplot(diamonds, aes(x=carat, y=price)) + theme_bw() + stat_bin_hex()

#data.table
# also check out dplyr
library(data.table)
data("diamonds")
diamonds <- data.table(diamonds)
class(diamonds)
diamonds[carat == 1, list(price, color)]
#SELECT price, color FROM diamonds WHERE carat = 1
#diamonds[WHERE clause, SELECT clause]

diamonds[carat == 1, mean(price)]
#SELECT avg(price) FROM diamonds WHERE carat = 1
diamonds[cut == 'Premium' & carat == 1, mean(price)]

#selecting columns based on a variable
my_columns <- c('price', 'color')
diamonds[carat == 1, my_columns, with=FALSE]

#Grouping
diamonds[,mean(price), by='cut']
#SELECT avg(price) FROM diamonds GROUP BY cut
#diamonds[WHERE clause, SELECT clause, by = GROUP BY CLAUSE]

diamonds[,mean(price), by=c('cut', 'color')][order(V1),]
#SELECT avg(price) FROM diamongs GROUP BY cut, color

diamonds[,cor(price, carat), by=c('cut')]

diamonds[,list(cr=cor(price, carat)), by=c('cut')][order(cr),]

diamonds[,list(cr=cor(price, carat), .N), by=list(round(carat))]

str(diamonds)
diamonds[,volume := x*y*z]
diamonds[,color := as.character(color)]
diamonds[,clarity := as.character(clarity)]
head(diamonds)
diamonds[carat < 2,as.list(coef(lm(price ~ carat + color))), by=c('cut')]

cf <- diamonds[,as.list(coef(lm(price ~ carat))), by=c('cut')]
setnames(cf, 'carat', 'carat_cf')
cf

#Inner join
joined <- merge(diamonds, cf, by='cut')

#Left join
joined <- merge(diamonds, cf, by='cut', all.x=TRUE)

#Right join
joined <- merge(diamonds, cf, by='cut', all.y=TRUE)

#Outer join
joined <- merge(diamonds, cf, by='cut', all.x=TRUE, all.y=TRUE)

#Get and set working directory
getwd()
setwd('~/workspace/data-science-scripts/zach/ODSC_2017/')

# A real example
dat <- fread('RegularSeasonCompactResults.csv')
teams <- fread('Teams.csv')

#Merge the winner's names
dat <- merge(dat, teams, by.x='Wteam', by.y='Team_Id', all.x=TRUE)
dat <- merge(
  dat, teams, by.x='Lteam', by.y='Team_Id', all.x=TRUE,
  suffixes = c('.w', '.l'))

#Merge the winner's names: Method 2
# dat <- merge(dat, teams, by.x='Wteam', by.y='Team_Id', all.x=TRUE)
# setnames(dat, 'Team_Name', 'Wteamname')
# dat <- merge(dat, teams, by.x='Lteam', by.y='Team_Id', all.x=TRUE)
# setnames(dat, 'Team_Name', 'Lteamname')

#Make the data a little more symmetric
dat[,Lloc := 'N']
dat[Wloc == 'H', Lloc := 'A']
dat[Wloc == 'A', Lloc := 'H']
dat[,table(Wloc, Lloc)]

#Make the data symmetric
winners <- dat[,list(
  Season, Daynum,
  team1 = Team_Name.w, team2 = Team_Name.l,
  score1 = Wscore, score2 = Lscore,
  loc = Wloc
)]
losers <- dat[,list(
  Season, Daynum,
  team1 = Team_Name.l, team2 = Team_Name.w,
  score1 = Lscore, score2 = Wscore,
  loc = Lloc
)]
dat_clean <- rbind(winners, losers, fill=TRUE)
head(winners, 1)
head(losers, 1)

#Subset to 2017
dat_clean <- dat_clean[Season == 2017,]
setkeyv(dat_clean, c('Daynum', 'score1', 'score2'))
dat_clean

#Which team is the best team
dat_clean[,point_diff := score1 - score2]
dat_summary <- dat_clean[,list(
  med = median(point_diff),
  mean = mean(point_diff),
  sd = sd(point_diff)
), by='team1']
setorder(dat_summary, -med)
head(dat_summary, 10)

#Lets fit a model
hist(dat_clean$point_diff)
model <- lm(point_diff ~ 0 + team1 + team2, dat_clean)
cf <- coef(summary(model))
cf <- data.table(
  var = row.names(cf),
  cf
  )
setorder(cf, -Estimate)
head(cf, 10)
tail(cf, 10)

#Lets do some machine learning
library(caret)
X <- dat_clean[,list(team1, team2)]
y <- dat_clean$point_diff
set.seed(42)
myControl <- trainControl(method='cv', number=5)
model_lm <- train(X, y, method='lm', trControl=myControl)
model_lm

#Fit a random forest using the same setup
set.seed(42)
myControl <- trainControl(method='cv', number=5)
model_rf <- train(X, y, method='ranger', trControl=myControl)
model_rf

#Compare 2 models
compare <- resamples(list(lm=model_lm, rf=model_rf))
dotplot(compare, metric='RMSE')


