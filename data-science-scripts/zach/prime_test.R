library(data.table)
library(bit64)
library(ggplot2)
library(scales)
library(reshape2)

#Load data
dat <- fread('~/datasets/PRIME_direct_mbtest_export.csv')
setnames(dat, make.names(names(dat), unique=TRUE))
dat <- dat[
  Sample_Pct == 64, list(
    #Model info
    lid=X_id,
    pid,
    Sample_Size,
    reference_model,
    is_blender,
    is_prime,
    parent_id,
    main_task,
    main_args,
    tasks=X_tasks,

    #Model stats
    Max_RAM_GB = Max_RAM / 1e+9,
    Total_Time_P1_mins = Total_Time_P1/60,
    holdout_scoring_rows_per_sec = holdout_size/holdout_scoring_time,
    max_vertex_storage_size_P1,

    # Project metric + error
    gs_metric,
    error_H,

    # Holdout error
    RMSE_H,
    LogLoss_H,
    Gini.Norm_H
  )]

#Split prime vs non prime
prime <- dat[is_prime == TRUE,]
non_prime <- dat[is_prime == FALSE,]
non_prime[,is_prime := NULL]

#Join to parent
prime <- prime[,list(
  lid = gsub("']", "", gsub("[u'", '', parent_id, fixed=TRUE), fixed=TRUE),
  Max_RAM_GB,
  Total_Time_P1_mins,
  holdout_scoring_rows_per_sec,
  max_vertex_storage_size_P1,
  gs_metric,
  RMSE_H,
  LogLoss_H,
  Gini.Norm_H
)]
keys <- c('lid')
others <- setdiff(names(prime), keys)
setnames(prime, others, paste0(others, '_prime'))

dat <- merge(non_prime, prime, by='lid')

#Clean data
dat[,best_model := Gini.Norm_H == max(Gini.Norm_H), by='pid']
dat[,best_prime := Gini.Norm_H_prime == max(Gini.Norm_H_prime), by='pid']
dat[,model_family := 'Unkown']
dat[grepl('^ASVM', main_task), model_family := "SVM"]
dat[grepl('^SVM', main_task), model_family := "SVM"]
dat[grepl('^GLM', main_task), model_family := "Linear"]
dat[grepl('^LR', main_task), model_family := "Linear"]
dat[grepl('^BENET', main_task), model_family := "Linear"]
dat[grepl('^BLENET', main_task), model_family := "Linear"]
dat[grepl('^ENET', main_task), model_family := "Linear"]
dat[grepl('^LENET', main_task), model_family := "Linear"]
dat[grepl('^ET', main_task), model_family := "Linear"] #Not sure
dat[grepl('^SGD', main_task), model_family := "Linear"]
dat[grepl('^RULEFIT', main_task), model_family := "Rulefit"]
dat[grepl('^RFC$', main_task), model_family := "RF"]
dat[grepl('^RFR$', main_task), model_family := "RF"]
dat[grepl('^RR$', main_task), model_family := "RF"] #Not sure
dat[grepl('^PGB', main_task), model_family := "GBM"]
dat[grepl('^GB', main_task), model_family := "GBM"]
dat[grepl('^XGB', main_task), model_family := "GBM"]
dat[grepl('^ESXGB', main_task), model_family := "GBM"]
dat[grepl('^ESGB', main_task), model_family := "GBM"]
dat[grepl('^PXGB', main_task), model_family := "GBM"]
dat[grepl('^MNGB', main_task), model_family := "GBM"]
dat[grepl('^WNGEC', main_task), model_family := "Text"]
dat[grepl('^CNBC', main_task), model_family := "Text"]
dat[grepl('^CNGEC', main_task), model_family := "Text"]
dat[grepl('^WNGER', main_task), model_family := "Text"]
dat[grepl('^DT', main_task), model_family := "Tree"]
dat[grepl('^TWSL2', main_task), model_family := "Tree"] #Not sure
dat[grepl('^KNN', main_task), model_family := "KNN"]
dat[is_blender == TRUE, model_family := "Blender"]

dat[,model_type := "Regular"]
dat[is_blender == TRUE, model_type := "Blender"]
dat[reference_model == TRUE,model_type := "Reference"]

#color scale:
#http://colorbrewer2.org/
colors <- c(
  "#1f78b4", "#ff7f00", "#6a3d9a", "#33a02c", "#e31a1c", "#b15928",
  "#a6cee3", "#fdbf6f", "#cab2d6", "#b2df8a", "#fb9a99", "#ffff99", "black",
  "grey", "grey"
)

#################################################
# Plot by model
#################################################

x <- dat[,table(diff > .90)]
x / sum(x)

#Plot 1 - Gini
ggplot(dat,aes(
  x=Gini.Norm_H,
  y=Gini.Norm_H_prime,
  col=model_family
)) +
  geom_abline(slope=1, intercept=0, alpha=.50, linetype=2) +
  geom_point(alpha=.50) +
  theme_bw() + theme(legend.position="bottom") +
  scale_color_manual(values=colors)

#Plot 2 - Gini
ggplot(dat,aes(
  x=Gini.Norm_H,
  y=Gini.Norm_H_prime,
  col=model_type
)) +
  geom_abline(slope=1, intercept=0, alpha=.50, linetype=2) +
  geom_point(alpha=.50) +
  theme_bw() + theme(legend.position="bottom") +
  scale_color_manual(values=colors) +
  facet_wrap(~model_family)

#Plot 3 - RMSE
ggplot(dat,aes(
  x=RMSE_H,
  y=RMSE_H_prime,
  col=model_family
)) +
  geom_abline(slope=1, intercept=0, alpha=.50, linetype=2) +
  geom_point(alpha=.50) +
  theme_bw() + theme(legend.position="bottom") +
  scale_color_manual(values=colors)

#Plot 4 - RMSE
ggplot(dat,aes(
  x=RMSE_H,
  y=RMSE_H_prime,
  col=model_type
)) +
  geom_abline(slope=1, intercept=0, alpha=.50, linetype=2) +
  geom_point(alpha=.50) +
  theme_bw() + theme(legend.position="bottom") +
  scale_color_manual(values=colors) +
  facet_wrap(~model_family)

#Plot 5 - LogLoss
ggplot(dat,aes(
  x=LogLoss_H,
  y=LogLoss_H_prime,
  col=model_family
)) +
  geom_abline(slope=1, intercept=0, alpha=.50, linetype=2) +
  geom_point(alpha=.50) +
  theme_bw() + theme(legend.position="bottom") +
  scale_color_manual(values=colors)

#Plot 6 - LogLoss
ggplot(dat,aes(
  x=LogLoss_H,
  y=LogLoss_H_prime,
  col=model_type
)) +
  geom_abline(slope=1, intercept=0, alpha=.50, linetype=2) +
  geom_point(alpha=.50) +
  theme_bw() + theme(legend.position="bottom") +
  scale_color_manual(values=colors) +
  facet_wrap(~model_family)

#################################################
# Plot prime (error)
#################################################

dat[,Gini.Norm_H_ref := max(Gini.Norm_H[reference_model], na.rm=TRUE),by='pid']

dat[, model_vs_prime := (Gini.Norm_H - Gini.Norm_H_prime) / Gini.Norm_H]
dat[, model_vs_ref := (Gini.Norm_H - Gini.Norm_H_ref) / Gini.Norm_H]

ggplot(dat,aes(
  x=model_vs_prime,
  y=model_vs_ref,
  col=model_type
)) +
  geom_abline(slope=1, intercept=0, alpha=.50, linetype=2) +
  geom_point(alpha=.50) +
  theme_bw() + theme(legend.position="bottom") +
  scale_color_manual(values=colors) +
  facet_wrap(~model_family)

ggplot(dat,aes(
  x=model_type,
  y=model_vs_prime,
  col=model_type
)) +
  geom_abline(slope=1, intercept=0, alpha=.50, linetype=2) +
  geom_violin(alpha=.50) +
  theme_bw() + theme(legend.position="bottom") +
  scale_color_manual(values=colors) +
  facet_wrap(~model_family)

#################################################
# Plot by project
#################################################

#Prime vs best vs ref
keys <- c('pid', 'Sample_Size')
proj <- dat[,list(
  Gini.Norm_H = max(Gini.Norm_H, na.rm=TRUE),
  Gini.Norm_H_prime = max(Gini.Norm_H_prime, na.rm=TRUE),
  Gini.Norm_H_ref = max(Gini.Norm_H[reference_model], na.rm=TRUE),
  RMSE_H = min(RMSE_H, na.rm=TRUE),
  RMSE_H_prime = min(RMSE_H_prime, na.rm=TRUE),
  RMSE_H_ref = min(RMSE_H[reference_model], na.rm=TRUE),
  LogLoss_H = min(LogLoss_H, na.rm=TRUE),
  LogLoss_H_prime = min(LogLoss_H_prime, na.rm=TRUE),
  LogLoss_H_ref = min(LogLoss_H[reference_model], na.rm=TRUE)
), by=keys]

#Reshape and replot
proj <- melt.data.table(proj, id.vars=keys)

proj[,metric := 'UNK']
proj[grepl('^Gini.Norm', variable), metric := 'Gini.Norm']
proj[grepl('^RMSE', variable), metric := 'RMSE']
proj[grepl('^LogLoss', variable), metric := 'LogLoss']

proj[,model := 'regular']
proj[grepl('_prime$', variable), model := 'prime']
proj[grepl('_ref$', variable), model := 'ref']

ggplot(proj[metric!='RMSE',],aes(
  x=model,
  color=model,
  y=value
)) +
  geom_boxplot() +
  theme_bw() + theme(legend.position="bottom") +
  scale_color_manual(values=colors) +
  facet_wrap(~metric,scales='free')


