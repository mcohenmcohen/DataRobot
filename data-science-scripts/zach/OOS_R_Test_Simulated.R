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
lb_base <- loadLeaderboard('5609cf39c8e2f429efa6249b')
lb_old <- loadLeaderboard('561559cb3a822e6463a652c2')
lb_new <- loadLeaderboard('561d63fa1058be7bca187d8f')
lb_base$run <- 'Reference OSS'
lb_old$run <- 'Old OSS'
lb_new$run <- 'Big Data OSS'

#Subset data
#dat <- copy(lb)
dat <- copy(rbindlist(list(lb_base, lb_old, lb_new), fill=TRUE))
length(unique(dat$Filename))
setnames(dat, make.names(names(dat)))
dat <- dat[,list(
  reference_model,
  main_task,
  is_blender,
  is_prime,
  dataset_size,
  Sample_Size,
  Sample_Pct,
  model_name,
  Y_Type,
  Filename,
  Project_Metric,
  Total_Time_P1,
  Total_Time_P2,
  Total_Time_P3,
  Total_Time_P4,
  Total_Time_P5,
  Max_RAM_GB,
  Cached_Time_P1,
  RMSE_H,
  holdout_scoring_time,
  error_P1,
  Gini.Norm_H,
  X_Rows,
  X_Cols,
  holdout_size,
  run='')]

#MOVE OVER TO shrinkR package
dat[,holdout_time_seconds := holdout_scoring_time/1000]

dat[,holdout_time_mins := holdout_time_seconds / 60]
dat[, Total_Time_Mins_P1to5 :=
      (Total_Time_P1 + Total_Time_P2 + Total_Time_P3 + Total_Time_P4 + Total_Time_P5) / 60]

#Fix names
#NEED TO LOOKUP IN MONGO!
#TODO: ADD RMONGO CONNECTIONS
dat <- dat[,model_name := gsub('^BLENDER w ', '', model_name)]
dat[, c('model', 'version', 'OSS') := list('', 0L, 'No')]

#Old OSS Models - old run only
dat <- dat[grepl('56154f99992214579a541a8b', model_name), c('model', 'version', 'OSS', 'run') := list('RF',  2L, 'Yes', 'V1.1')]
dat <- dat[grepl('56154f99992214579a541a8f', model_name), c('model', 'version', 'OSS', 'run') := list('RF', 2L, 'Yes', 'V1.1')]
dat <- dat[grepl('56154f99992214579a541a8d', model_name), c('model', 'version', 'OSS', 'run') := list('GBM',  2L, 'Yes', 'V1.1')]
dat <- dat[grepl('56154f99992214579a541a91', model_name), c('model', 'version', 'OSS', 'run') := list('GBM', 2L, 'Yes', 'V1.1')]

#"Big Data" OSS models - new run only
dat <- dat[grepl('56157847100d2b51b4765b52', model_name), c('model', 'version', 'OSS', 'run') := list('RF',  2L, 'Yes', 'V2')]
dat <- dat[grepl('56157847100d2b51b4765b54', model_name), c('model', 'version', 'OSS', 'run') := list('RF', 2L, 'Yes', 'V2')]
dat <- dat[grepl('56157847100d2b51b4765b56', model_name), c('model', 'version', 'OSS', 'run') := list('GBM',  2L, 'Yes', 'V2')]
dat <- dat[grepl('56157847100d2b51b4765b58', model_name), c('model', 'version', 'OSS', 'run') := list('GBM', 2L, 'Yes', 'V2')]

#Big data OSS models, clean/dirty big/little
dat <- dat[grepl('56157847100d2b51b4765b52', model_name), c('model', 'version', 'OSS', 'run') := list('RF',  2L, 'Yes', 'V2')]
dat <- dat[grepl('56157847100d2b51b4765b54', model_name), c('model', 'version', 'OSS', 'run') := list('RF', 2L, 'Yes', 'V2')]
dat <- dat[grepl('56157847100d2b51b4765b56', model_name), c('model', 'version', 'OSS', 'run') := list('GBM',  2L, 'Yes', 'V2')]
dat <- dat[grepl('56157847100d2b51b4765b58', model_name), c('model', 'version', 'OSS', 'run') := list('GBM', 2L, 'Yes', 'V2')]

dat <- dat[grepl('5616a0ff9922144798ccc167', model_name), c('model', 'version', 'OSS', 'run') := list('RF',  2L, 'Yes', 'V1.2')]
dat <- dat[grepl('5616a0ff9922144798ccc16b', model_name), c('model', 'version', 'OSS', 'run') := list('RF', 2L, 'Yes', 'V1.2')]
dat <- dat[grepl('5616a0ff9922144798ccc169', model_name), c('model', 'version', 'OSS', 'run') := list('GBM',  2L, 'Yes', 'V1.2')]
dat <- dat[grepl('5616a0ff9922144798ccc16d', model_name), c('model', 'version', 'OSS', 'run') := list('GBM', 2L, 'Yes', 'V1.2')]

#Lookit some stuff
dat[is_blender == FALSE,table(model, version)]
dat[OSS=='Yes', table(model, Y_Type)]
dat[OSS=='Yes', table(model, Y_Type, is.na(Total_Time_P1))]
dat[OSS=='Yes', table(model, Y_Type, is.na(Gini.Norm_H))]

#Baseline versions of models
dat <- dat[main_task == 'RGBR', c('model', 'run') := list('GBM',  'RV0')]
dat <- dat[main_task == 'RGBC', c('model', 'run') := list('GBM',  'RV0')]
dat <- dat[main_task == 'GBC' & reference_model==T, c('model', 'run') := list('GBM',  'PV0')]
dat <- dat[main_task == 'GBR' & reference_model==T, c('model', 'run') := list('GBM',  'PV0')]

dat <- dat[main_task == 'RRFC2', c('model', 'run') := list('RF',  'RV0')]
dat <- dat[main_task == 'RRFR2', c('model', 'run') := list('RF',  'RV0')]
dat <- dat[main_task == 'RFC' & reference_model==T, c('model', 'run') := list('RF',  'PV0')]
dat <- dat[main_task == 'RFR' & reference_model==T, c('model', 'run') := list('RF',  'PV0')]

#Lookup 0 gini
dat[OSS!='No' & Gini.Norm_H==0,]

#1-way
id_vars <- c('run', 'Sample_Size', 'model_name', 'Y_Type', 'Filename', 'X_Rows', 'X_Cols', 'model', 'OSS', 'holdout_size')
measure_vars <- c('Total_Time_P1', 'holdout_time_seconds','Gini.Norm_H', 'Max_RAM_GB')
x <- copy(dat[is_blender==FALSE,])
x <- melt(x[,c(id_vars, measure_vars), with=FALSE], measure.vars=measure_vars)
x <- x[!is.na(value),]
sort(table(x$model))
ggplot(x, aes(
  x=X_Rows * X_Cols,
  #x=holdout_size * X_Cols,
  y=value,
  col=OSS, shape=Y_Type
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
