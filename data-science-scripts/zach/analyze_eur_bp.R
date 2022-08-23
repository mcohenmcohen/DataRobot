stop()
rm(list=ls(all=T))
gc(reset=T)
library(yaml)
library(data.table)
library(readr)
library(httr)
library(pbapply)
library(stringi)
library(ggplot2)
library(scales)
library(viridis)
library(ggthemes)

# Load Raw data
baseline_raw <- fread('http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests=598c7bb17423161098cfbb50&max_sample_size_only=false')
eureqa_raw <- fread('http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests=59ceb01a4974e3000163477c&max_sample_size_only=false')

# Combine the 2 datasets
baseline <- copy(baseline_raw)
eureqa <- copy(eureqa_raw)

baseline[,blueprint := 'Auto']
eureqa[,blueprint := 'Eureqa']
dat <- rbind(baseline, eureqa, fill=T)

# Format some units
dat[,Max_RAM_GB := as.numeric(Max_RAM) * 1e-9]
dat[,Total_Time_P1_Hours := Total_Time_P1/3600]
dat[,size_GB := size * 1e-9]
dat[,Sample_Pct := as.numeric(Sample_Pct)]
dat[,table(Sample_Pct)]

# Lookup
dat[blueprint == 'Auto' & Sample_Pct <= 90 & Total_Time_P1_Hours > 2,`_id`]


# Order the metrics
dat[, metric := gsub('Weighted ', '', metric)]
dat[,sort(unique(metric))]
dat[,metric := factor(metric, levels=c('RMSE', 'LogLoss', 'AUC', 'Gamma Deviance', 'Poisson Deviance', 'Tweedie Deviance'))]

# Classify projects
dat[,type := 'ERROR']
dat[metric %in% c('RMSE', 'Gamma Deviance', 'Poisson Deviance', 'Tweedie Deviance'), type := 'Reg']
dat[metric %in% c('LogLoss', 'AUC'), type := 'Class']
dat[,table(metric, type)]

# Subset to datasets we have for both mbtests
files_eur <- dat[!is.na(`Gini Norm_P1`) & blueprint == 'Auto', sort(unique(Filename))]
files_auto <- dat[!is.na(`Gini Norm_P1`) & blueprint == 'Eureqa', sort(unique(Filename))]
files <- intersect(files_eur, files_auto)
dat <- dat[Filename %in% files,]

# Aggregate autopilot stats
keys <- c('Filename', 'blueprint', 'metric')
agg <- dat[
  Sample_Pct <= 70,
  list(
    gini_v = max(`Gini Norm_P1`, na.rm=T),
    gini_h = max(`Gini Norm_P1`, na.rm=T),
    hours = sum(Total_Time_P1_Hours, na.rm=T),
    # Sample_Pct = max(Sample_Pct, na.rm=T),
    Max_RAM_GB = max(Max_RAM_GB, na.rm=T)
  ), by=keys]
agg <- melt.data.table(agg, id.vars=keys)

# Custom color scale
colors <- c(
  "#1f78b4", "#ff7f00", "#6a3d9a", "#33a02c", "#e31a1c", "#b15928",
  "#a6cee3", "#fdbf6f", "#cab2d6", "#b2df8a", "#fb9a99", "black",
  "grey"
)

# Scatterplots
plotdat <- dcast.data.table(agg, Filename + variable + metric ~ blueprint)
plotdat[Eureqa > Auto & variable %in% c('gini_v', 'gini_h'),]
ggplot(plotdat, aes(x=Eureqa, y=Auto, color=metric)) +
  geom_point() + geom_abline(slope=1) +
  scale_color_manual(values=colors) +
  theme_tufte(base_family="Helvetica") +
  theme(legend.position="bottom") +
  facet_wrap(~variable, scales='free') +
  ggtitle('Eureqa vs Best Autopilot Model')

# Boxplots
plotdat <- copy(agg)
variables_to_log <- c('hours', 'Max_RAM_GB')
plotdat[variable %in% variables_to_log, value := log10(1+value)]
plotdat[variable %in% variables_to_log, variable := paste0('log10_', variable)]
ggplot(plotdat, aes(x=blueprint, y=value, color=blueprint)) +
  geom_boxplot() +
  scale_color_manual(values=colors) +
  theme_tufte(base_family="Helvetica") +
  theme(legend.position="bottom") +
  facet_wrap(~variable, scales='free') +
  ggtitle('Eureqa vs Best Autopilot Model')

# Prediction stats
keys <- c('Filename', 'blueprint', 'metric', 'type')
agg_pred <- dat[
  Sample_Pct > 90,
  list(
    gini_p = max(`Prediction Gini Norm`, na.rm=T),
    logloss_p = min(`Prediction LogLoss`, na.rm=T),
    rmse_p = min(`Prediction RMSE`, na.rm=T),
    mad_p = max(`Prediction MAD`, na.rm=T)
  ), by=keys]
agg_pred <- melt.data.table(agg_pred, id.vars=keys)
#agg_pred[! metric %in% c('AUC', 'LogLoss') & variable %in% 'logloss_p', value := NA]
#agg_pred[ metric %in% c('AUC', 'LogLoss') & variable %in% c('rmse_p', 'mad_p'), value := NA]
agg_pred <- agg_pred[is.finite(value),]

# Scatterplots
plotdat <- copy(agg_pred)
plotdat[type == 'Reg' & variable %in% c('logloss_p', 'auc_p'), value := NA]
plotdat[type == 'Class' & !variable %in% c('logloss_p', 'auc_p', 'gini_p'), value := NA]
variables_to_log <- c('rmse_p', 'mad_p')
plotdat[variable %in% variables_to_log, value := log10(1+value)]
plotdat[variable %in% variables_to_log, variable := paste0('log10_', variable)]
plotdat <- plotdat[!is.na(value),]
plotdat <- dcast.data.table(plotdat, Filename + variable + metric ~ blueprint)
ggplot(plotdat, aes(x=Eureqa, y=Auto, color=metric)) +
  geom_point() + geom_abline(slope=1) +
  scale_color_manual(values=colors) +
  theme_tufte(base_family="Helvetica") +
  theme(legend.position="bottom") +
  facet_wrap(~variable, scales='free') +
  ggtitle('Eureqa vs Best Autopilot Model - prediction datasets')

# Long running EUR Bps
agg[blueprint == 'Eureqa' & variable == 'hours' & value > 1,][order(value),dput(Filename)]
