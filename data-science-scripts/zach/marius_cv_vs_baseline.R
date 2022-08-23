#Load Data
library('shrinkR')
library('data.table')
library('reshape2')
library('ggplot2')
library('scales')

#Useful mongo commands:
# use MMApp
# db.open_model_code.find({}, {name:1, version:1, autopilot_version:1})

#MBtest IDs
# 55f201fa1058be218a1909c6 - 202 all models, somehow skipped OSS models...
# 55f30dad3a822e21029a425f - 202 VS OSS
# 55f2067e3a822e3ee83427f2 - 202 V3 OSS
# 55f81bcf3a822e5b898beeb4 - simulated data
# 56099ea53a822e5a97482fa4 - simulated data with blender fix, OSS only
# 5609cf39c8e2f429efa6249b - simulated data with blender fix, all models
# 560e8e5c87d19e78b47098e6 - bigger simulated data, blender fix, NEW OSS only
# 561579f43a822e4b9dbf129b - bigger simulated clean/dirty, reference and old OSS only - missing timings
# 561559cb3a822e6463a652c2 - bigger simulated clean/dirty, old OSS only
# 561d63fa1058be7bca187d8f - Bigger simulated data, should have old and new oss

#Mongo Code:
# mongo --host=mongo-0.mbtest.us-east-1.aws.datarobot.com
# use MMApp
# db.open_model_code.find({},{_id:1, name:1, version:1, autopilot_versions:1, created:1}).sort({created:1})

#Push new models to remote mongo:
#python ModelingMachine/metablueprint/oss_models.py --host="mongo-0.mbtest.us-east-1.aws.datarobot.com" --R_dir=../open-source-scripts/R --override-engine

#check <- verifyYaml('~/workspace/datarobot/tests/ModelingMachine/out_of_core_rulefit_datasets_largedata_regonly.yaml')

#Load base MBtest and new run
lb_base <- loadLeaderboard('57173b19dde5c4530e82737a')
lb_new <- loadLeaderboard('57168fd4746ddc533b14fa0c')

lb_base$id <- '57173b19dde5c4530e82737a'
lb_new$id <- '57168fd4746ddc533b14fa0c'

lb_base$run <- 'nightly_master'
lb_new$run <- 'marius_pr'

#Subset data
dat <- rbindlist(list(lb_base, lb_new), fill=TRUE)
length(unique(dat$Filename))
setnames(dat, make.names(names(dat)))
dat[,Blueprint := sapply(Blueprint, paste, collapse=' ')]
dat[,Task_Info_Extras := sapply(Task_Info_Extras, paste, collapse=' ')]
dat <- dat[,list(
  run,
  metric,
  Filename,
  is_blender,
  metric,
  model_name,
  main_task,
  Blueprint,
  Sample_Pct,
  cv_method,
  Total_Time_P1,
  Max_RAM_GB,
  holdout_scoring_time,
  Gini.Norm_H,
  Task_Info_Extras,
  cv_method)]
dat <- unique(dat)
dat[Filename == 'forestfires.csv',]

#MOVE OVER TO shrinkR package
dat[,holdout_time_seconds := holdout_scoring_time/1000]
dat[,holdout_scoring_time := NULL]

#Reshape
id_vars <- c("run", "Filename", "model_name", "main_task", "Blueprint", "Sample_Pct", "cv_method", "metric", "is_blender")
measure_vars <- c('Total_Time_P1', 'holdout_time_seconds', 'Gini.Norm_H', 'Max_RAM_GB')
x <- melt.data.table(dat[,c(id_vars, measure_vars), with=FALSE], measure.vars=measure_vars)
x <- x[!is.na(value),]

cols <- c('run')
rows <- setdiff(c(id_vars, 'variable'), cols)
flma <- as.formula(paste(paste(rows, collapse=' + '), paste(cols, collapse=' + '), sep=' ~ '))
x <- dcast.data.table(x, flma)
x[marius_pr >1,]

#1-way

x <- copy(dat[is_blender==FALSE,])

ggplot(x, aes(
  y=value,
  col=id, shape=Y_Type
)) +
  geom_point() +
  theme_bw() +
  scale_colour_manual(values=c('#984ea3', '#4daf4a', '#377eb8', '#ff7f00', '#e41a1c')) +
  theme(legend.position = "bottom") +
  facet_wrap(~variable, scales='free', ncol=2) +
  scale_x_log10(labels = comma)  +
  scale_y_log10(labels = comma, breaks=c(0.001,.1,1,10,100,1000,10000)) + expand_limits(x = 0, y = 0) + xlab('Dataset Rows X Columns') +
  xlab('Dataset Rows X Columns') +
  ylab('')

#2-way
# yp0 <- x[run=='PV0' & Filename=='2006_2010_Draft_Tools_for_is50p.csv', list(Sample_Size, X_Rows, X_Cols, Y_Type, Filename, model, run, variable, value)]
# yr0 <- x[run=='RV0' & Filename=='2006_2010_Draft_Tools_for_is50p.csv', list(Sample_Size, X_Rows, X_Cols, Y_Type, Filename, model, run, variable, value)]
# y2 <- x[run=='V2' & Filename=='2006_2010_Draft_Tools_for_is50p.csv', list(Sample_Size, X_Rows, X_Cols, Y_Type, Filename, model, run, variable, value)]
# y3 <- x[run=='V3' & Filename=='2006_2010_Draft_Tools_for_is50p.csv', list(Sample_Size, X_Rows, X_Cols, Y_Type, Filename, model, run, variable, value)]
# y <- rbindlist(list(yp0, yr0, y2, y3))
# y[order(Filename, Sample_Size, X_Rows, X_Cols, Y_Type, model, run, variable, value),]
# dcast.data.table(y,  Sample_Size + X_Rows + X_Cols + Y_Type + Filename + model + variable ~ run)

y <- dcast.data.table(
  x[run %in% c('PV0', 'RV0', 'V1.1', 'V1.2', 'V1.3', 'V1.4', 'V2'),],
  X_Rows + X_Cols + Y_Type + Filename + model + variable  ~ run)
#y <- y[!is.na(V2),]

#R V0 vs V1
#size=log10(X_Rows * X_Cols)
y[,V1 := V1.2]
y[!is.na(RV0) & variable == 'holdout_time_seconds', summary(V1/RV0)]
ggplot(y[!is.na(RV0),], aes(x=RV0, y=V1, col=model, shape=Y_Type)) +
  geom_point() +
  theme_bw() +
  scale_colour_manual(values=c('#ff7f00', '#377eb8', '#984ea3', '#4daf4a', '#e41a1c')) +
  theme(legend.position = "bottom") +
  geom_abline(slope=1, linetype = 2) +
  facet_wrap(~variable, scales='free', ncol=2) +
  coord_fixed() + expand_limits(x = 0, y = 0) +
  xlab('R Reference Models') +
  ylab('R OSS Models') +
  ggtitle('R Reference models vs R open source models')

#R V1 vs V2
#size=log10(X_Rows * X_Cols)
y$V1 <- y$V1.2
ggplot(y[!is.na(V1),], aes(x=V1, y=V2, col=model, shape=Y_Type)) +
  geom_point() +
  theme_bw() +
  scale_colour_manual(values=c('#ff7f00', '#377eb8', '#984ea3', '#4daf4a', '#e41a1c')) +
  theme(legend.position = "bottom") +
  geom_abline(slope=1, linetype = 2) +
  facet_wrap(~variable, scales='free', ncol=2) +
  coord_fixed() + expand_limits(x = 0, y = 0) +
  xlab('R OSS Models') +
  ylab('R Big Data Models') +
  ggtitle('R open source models vs big-data')

#Python V0 vs V2
ggplot(y[!is.na(PV0),], aes(x=PV0, y=V2, col=model, shape=Y_Type)) +
  geom_point() +
  theme_bw() +
  scale_colour_manual(values=c('#ff7f00', '#377eb8', '#984ea3', '#4daf4a', '#e41a1c')) +
  theme(legend.position = "bottom") +
  geom_abline(slope=1, linetype = 2) +
  facet_wrap(~variable, scales='free', ncol=2) +
  coord_fixed() + expand_limits(x = 0, y = 0) +
  xlab('Python Reference Models') +
  ylab('R OSS Models')
