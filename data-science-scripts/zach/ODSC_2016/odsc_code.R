colors <- c(
  "#1f78b4", "#ff7f00", "#6a3d9a", "#33a02c", "#e31a1c", "#b15928",
  "#a6cee3", "#fdbf6f", "#cab2d6", "#b2df8a", "#fb9a99", "#ffff99", "black",
  "grey", "grey"
)

#Load dataset
library(data.table)
dat_raw <- fread("~/Downloads/odsc_football_modeling_data.csv")
dat <- dat_raw[week > 1,]
dat[full_name=='Rob Havenstein',]
dat <- dat[!is.na(prev_fanduel_points),]

#Lookit player point
players <- dat[,list(pts = sum(fanduel_points)),by='player_id']
hist(players$pts, breaks=100)
summary(players$pts)
nrow(players)
quantile(players$pts, 1:9/10)

#Filter out some players
players <- players[pts > 10,]
dat <- dat[player_id %in% players$player_id,]
dat[,table(position)]

#Initial scatterplot
library(ggplot2)
ggplot(dat, aes(x=mean_fanduel_points, y=fanduel_points)) +
  geom_point(alpha=0.25) +
  geom_smooth() +
  theme_bw()

#Scatter by week
ggplot(dat, aes(x=mean_fanduel_points, y=fanduel_points)) +
  geom_point(alpha=0.25) +
  geom_smooth() +
  facet_wrap(~week) +
  theme_bw()

#Plot correlation between mean and actual over time
cors <- dat[,list(
  cor=cor(
    mean_fanduel_points,
    fanduel_points,
    use='pairwise.complete.obs')
  ), by='week']
cors
ggplot(cors, aes(x=week, y=cor)) +
  geom_line() +
  theme_bw()

#Points by position
ggplot(dat, aes(x=position, y=fanduel_points)) +
  geom_boxplot() + theme_bw()

#QBs - boxplot
dat[,home := team == home_team]
ggplot(dat[position=='QB',], aes(x=home, y=fanduel_points)) +
  geom_boxplot() + theme_bw() + facet_wrap(~position)

#QBs - violinplot
ggplot(dat[position=='QB',], aes(x=home, y=fanduel_points)) +
  geom_violin() + theme_bw() + facet_wrap(~position)

#QB cor with fanduel points
numeric_vars <- names(dat)[sapply(dat, is.numeric)]
numeric_vars <- setdiff(numeric_vars, c('fanduel_points', 'week'))
QBs <- dat[position == 'QB',]
cors <- cor(
  QBs$fanduel_points,
  QBs[,numeric_vars,with=F],
  use='pairwise.complete.obs')[1,]
head(sort(cors, decreasing=T), 10)

#Split train/test
train_weeks <- 2:13
test_weeks <- 14:17
train_indexes <- dat[,which(week %in% train_weeks)]
test_indexes <- dat[,which(week %in% test_weeks)]
train_dat <- dat[train_indexes,]
test_dat <- dat[test_indexes,]

#Linear model
xvars <- c('position', 'home', numeric_vars)
xvars <- paste(xvars, collapse=' + ')
model_formula <- paste('fanduel_points ~ ', xvars)
model_formula <- as.formula(model_formula)
model_lm <- lm(model_formula, train_dat)
#summary(model)
cfs <- coef(summary(model_lm))
cfs <- cfs[order(cfs[,4]),]
round(head(cfs, 10), 4)

#Random forest model
library(ranger)
set.seed(42)
model_rf <- ranger(
  model_formula, train_dat,
  write.forest=TRUE,
  importance='permutation')
head(sort(importance(model_rf), decreasing=TRUE), 10)

#Out of sample error
act <- test_dat$fanduel_points
pred_lm <- predict(model_lm, test_dat)
pred_rf <- predict(model_rf, test_dat)$predictions
print(sqrt(mean((pred_lm-act)^2)))
print(sqrt(mean((pred_rf-act)^2)))
plot_dat <- data.frame(
  act,
  pred_lm,
  pred_rf
)

#Out of sample plots
ggplot(plot_dat, aes(x=pred_rf, y=act)) +
  geom_point(alpha=.25) + geom_smooth() +
  ggtitle('Out of sample predicted vs actual for a linear model') +
  theme_bw()
ggplot(plot_dat, aes(x=pred_lm, y=act)) +
  geom_point(alpha=.25) + geom_smooth() +
  ggtitle('Out of sample predicted vs actual for a random forest model') +
  theme_bw()
ggplot(plot_dat, aes(x=pred_rf, y=pred_lm)) +
  geom_point(alpha=.25) + geom_smooth() +
  ggtitle('Out of sample predictions for rf vs lm') +
  theme_bw()

#Mean points allowed by week
pts_per_week <- dat_raw[player_id %in% players$player_id,list(
  fanduel_points = sum(fanduel_points),
  n=.N),
  by=c('week', 'position', 'opponent')]
setkeyv(pts_per_week, c('position', 'opponent', 'week'))
pts_per_week[,cumul_pts := cumsum(fanduel_points), by=c('position', 'opponent')]
pts_per_week[,cumul_n := cumsum(n), by=c('position', 'opponent')]
pts_per_week[,mean_points_allowed := cumul_pts / cumul_n]
pts_per_week <- pts_per_week[,list(week, position, opponent, mean_points_allowed)]
pts_per_week[,week := week + 1]
ggplot(pts_per_week[
  opponent %in% c('NE', 'NYJ')
], aes(x=week, y=mean_points_allowed, col=position)) +
  geom_line() + theme_bw() +
  facet_wrap(~opponent)

#Non-rolling join
dat2 <- merge(dat, pts_per_week, by=c('position', 'opponent', 'week'), all.x=TRUE)
summary(dat2$mean_points_allowed)

#Add to original data
#Rolling join
keys <- c('position', 'opponent', 'week')
setkeyv(dat, keys)
setkeyv(pts_per_week, keys)
dat2 <- pts_per_week[dat,,roll=TRUE,rollends=TRUE]
summary(dat2$mean_points_allowed)

#Adjust by positional mean and week
position_means <- dat[,list(mean_points_by_pos = mean(fanduel_points)), by='position']
print(position_means)

#Merge back to dataset
dat2 <- merge(dat2, position_means, by='position')

#Split train/test
train_weeks <- 2:13
test_weeks <- 14:17
train_indexes <- dat2[,which(week %in% train_weeks)]
test_indexes <- dat2[,which(week %in% test_weeks)]
train_dat <- dat2[train_indexes,]
test_dat <- dat2[test_indexes,]

#Re run Linear model
numeric_vars <- names(dat2)[sapply(dat2, is.numeric)]
numeric_vars <- setdiff(numeric_vars, c('fanduel_points', 'week'))
xvars <- c('position', 'home', numeric_vars)
xvars <- paste(xvars, collapse=' + ')
model_formula <- paste('fanduel_points ~ ', xvars)
model_formula <- as.formula(model_formula)
model_lm <- lm(model_formula, train_dat)
#summary(model)
cfs <- coef(summary(model_lm))
cfs <- cfs[order(cfs[,4]),]
round(head(cfs, 10), 4)

#Re run Random forest model
library(ranger)
set.seed(42)
model_rf <- ranger(
  model_formula, train_dat,
  write.forest=TRUE,
  importance='permutation')
head(sort(importance(model_rf), decreasing=TRUE), 10)


#Re run Random forest model
library(glmnet)
set.seed(42)
x <- model.matrix(model_formula, train_dat)[,-1]
x_test <- model.matrix(model_formula, test_dat)[,-1]
model_glmnet <- cv.glmnet(x, train_dat[,fanduel_points], alpha=.99)

#Out of sample error
act <- test_dat$fanduel_points
pred_lm <- predict(model_lm, test_dat)
pred_glmnet <- predict(model_glmnet, x_test)
pred_rf <- predict(model_rf, test_dat)$predictions
print(sqrt(mean((pred_lm-act)^2)))
print(sqrt(mean((pred_glmnet-act)^2)))
print(sqrt(mean((pred_rf-act)^2)))
plot_dat <- data.frame(
  act,
  pred_lm,
  pred_rf
)

#Out of sample plots
ggplot(plot_dat, aes(x=pred_rf, y=act)) +
  geom_point(alpha=.25) + geom_smooth() +
  ggtitle('Out of sample predicted vs actual for a linear model') +
  theme_bw()
ggplot(plot_dat, aes(x=pred_lm, y=act)) +
  geom_point(alpha=.25) + geom_smooth() +
  ggtitle('Out of sample predicted vs actual for a random forest model') +
  theme_bw()
ggplot(plot_dat, aes(x=pred_rf, y=pred_lm)) +
  geom_point(alpha=.25) + geom_smooth() +
  ggtitle('Out of sample predictions for rf vs lm') +
  theme_bw()

