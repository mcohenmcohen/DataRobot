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

#check <- verifyYaml('~/workspace/datarobot/tests/ModelingMachine/out_of_core_rulefit_datasets_largedata_regonly.yaml')

#Load base MBtest and new run
lb_base <- loadLeaderboard('55f81bcf3a822e5b898beeb4')
lb_V2 <- loadLeaderboard('55f30dad3a822e21029a425f')
lb_V3 <- loadLeaderboard('55f2067e3a822e3ee83427f2')
lb_base[, run := 'base']
lb_V2[, run := 'V2']
lb_V3[, run := 'V3']

length(unique(lb_base$Filename))
length(unique(lb_V2$Filename))
length(unique(lb_V3$Filename))

#Subset data
#dat <- copy(rbindlist(list(lb_base, lb_V2, lb_V3), fill=TRUE))
dat <- copy(lb_base)
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
  run)]
dat[,holdout_time_mins := holdout_scoring_time / 60]
dat[, Total_Time_Mins_P1to5 :=
      (Total_Time_P1 + Total_Time_P2 + Total_Time_P3 + Total_Time_P4 + Total_Time_P5) / 60]

#Fix names
#NEED TO LOOKUP IN MONGO!
#TODO: ADD RMONGO CONNECTIONS
dat <- dat[,model_name := gsub('^BLENDER w ', '', model_name)]
dat[, c('model', 'version', 'OSS') := list('', 0L, 'No')]
dat <- dat[grepl('55f1e87e100d2b5c7a16be6a', model_name), c('model', 'version', 'OSS') := list('RF',  2L, 'V2')]
dat <- dat[grepl('55f1e87e100d2b5c7a16be6c', model_name), c('model', 'version', 'OSS') := list('RF',  2L, 'V2')]
dat <- dat[grepl('55f1e87e100d2b5c7a16be6e', model_name), c('model', 'version', 'OSS') := list('GBM', 2L, 'V2')]
dat <- dat[grepl('55f1e87e100d2b5c7a16be70', model_name), c('model', 'version', 'OSS') := list('GBM', 2L, 'V2')]
dat <- dat[grepl('55f1e887100d2b5eec98cb64', model_name), c('model', 'version', 'OSS') := list('RF',  3L, 'V3')]
dat <- dat[grepl('55f1e887100d2b5eec98cb66', model_name), c('model', 'version', 'OSS') := list('RF',  3L, 'V3')]
dat <- dat[grepl('55f1e887100d2b5eec98cb68', model_name), c('model', 'version', 'OSS') := list('GBM', 3L, 'V3')]
dat <- dat[grepl('55f1e887100d2b5eec98cb6a', model_name), c('model', 'version', 'OSS') := list('GBM', 3L, 'V3')]
dat[model == '', model := main_task]
dat[is_blender == FALSE,table(model, version)]

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
measure_vars <- c('Total_Time_Mins_P1to5', 'holdout_time_mins','Gini.Norm_H', 'Max_RAM_GB')
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
  scale_colour_manual(values=c('#ff7f00', '#377eb8', '#984ea3', '#4daf4a', '#e41a1c')) +
  theme(legend.position = "bottom") +
  facet_wrap(~variable, scales='free', ncol=2) +
  scale_x_log10(labels = comma)  +
  scale_y_log10(labels = comma, breaks=c(1,10,100,1000,10000))

#2-way
# yp0 <- x[run=='PV0' & Filename=='2006_2010_Draft_Tools_for_is50p.csv', list(Sample_Size, X_Rows, X_Cols, Y_Type, Filename, model, run, variable, value)]
# yr0 <- x[run=='RV0' & Filename=='2006_2010_Draft_Tools_for_is50p.csv', list(Sample_Size, X_Rows, X_Cols, Y_Type, Filename, model, run, variable, value)]
# y2 <- x[run=='V2' & Filename=='2006_2010_Draft_Tools_for_is50p.csv', list(Sample_Size, X_Rows, X_Cols, Y_Type, Filename, model, run, variable, value)]
# y3 <- x[run=='V3' & Filename=='2006_2010_Draft_Tools_for_is50p.csv', list(Sample_Size, X_Rows, X_Cols, Y_Type, Filename, model, run, variable, value)]
# y <- rbindlist(list(yp0, yr0, y2, y3))
# y[order(Filename, Sample_Size, X_Rows, X_Cols, Y_Type, model, run, variable, value),]
# dcast.data.table(y,  Sample_Size + X_Rows + X_Cols + Y_Type + Filename + model + variable ~ run)

y <- dcast.data.table(
  x[run %in% c('PV0', 'RV0', 'V2', 'V3'),],
  X_Rows + X_Cols + Y_Type + Filename + model + variable ~ run)
y <- y[!is.na(V2),]

#R V0 vs V2
#size=log10(X_Rows * X_Cols)
ggplot(y[!is.na(RV0),], aes(x=RV0, y=V2, col=model, shape=Y_Type)) +
  geom_point() +
  theme_bw() +
  scale_colour_manual(values=c('#ff7f00', '#377eb8', '#984ea3', '#4daf4a', '#e41a1c')) +
  theme(legend.position = "bottom") +
  geom_abline(slope=1, linetype = 2) +
  facet_wrap(~variable, scales='free', ncol=2) +
  coord_fixed()

#Python V0 vs V2
ggplot(y[!is.na(PV0),], aes(x=PV0, y=V2, col=model, shape=Y_Type)) +
  geom_point() +
  theme_bw() +
  scale_colour_manual(values=c('#ff7f00', '#377eb8', '#984ea3', '#4daf4a', '#e41a1c')) +
  theme(legend.position = "bottom") +
  geom_abline(slope=1, linetype = 2) +
  facet_wrap(~variable, scales='free', ncol=2) +
  coord_fixed()

#V2 vs V3
ggplot(y[!is.na(V3),], aes(x=V2, y=V3, col=model, shape=Y_Type, size=log10(X_Rows * X_Cols))) +
  geom_point() +
  theme_bw() +
  scale_colour_manual(values=c('#ff7f00', '#377eb8', '#984ea3', '#4daf4a', '#e41a1c')) +
  theme(legend.position = "bottom") +
  geom_abline(slope=1, linetype = 2) +
  facet_wrap(~variable, scales='free', ncol=2) +
  coord_fixed()
