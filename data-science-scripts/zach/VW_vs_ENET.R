#Load Data
library('shrinkR')
library('data.table')
library('reshape2')
library('ggplot2')
library('ggrepel')
library('scales')
library('jsonlite')
library('pbapply')
library('stringi')

#Load data
suppressWarnings(ENET <- getLeaderboard('579a3e0ca4d22170960c78d2'))
suppressWarnings(VW <- getLeaderboard('579a29ef2a595823f4860497'))
ENET[,test := 'ENET']
VW[,test := 'VW']

#Combine data
dat <- rbind(ENET, VW, fill=TRUE, use.names=TRUE)
setnames(dat, make.names(names(dat), unique=TRUE))
setnames(dat, make.names(names(dat), unique = TRUE))

#Remove 95% runs trained into holdout:
dat[,holdout_pct := round(holdout_size/(Sample_Size/(Sample_Pct/100)),3)]
summary(dat$holdout_pct); dim(dat)
dat <- dat[!is.na(holdout_pct),]
summary(dat$holdout_pct); dim(dat)
dat <- dat[holdout_pct < 0.05,]
summary(dat$holdout_pct); dim(dat)

#Subset
#dat[,table(Sample_Pct, test)]
#dat <- dat[Sample_Pct==95,]

#Remove prime models
dat <- dat[main_task != 'RULEFITR',]

#Clean Blueprints
dat[,Blueprint := sapply(Blueprint, function(x) paste(unlist(x), collapse=" "))]
dat[,Blueprint := stri_replace_all_fixed(Blueprint, "u'", '')]
dat[,Blueprint := stri_replace_all_fixed(Blueprint, "'", '')]
dat[,Blueprint := stri_replace_all_fixed(Blueprint, "[", ' ')]
dat[,Blueprint := stri_replace_all_fixed(Blueprint, "]", ' ')]
dat[,Blueprint := stri_replace_all_fixed(Blueprint, "{", ' ')]
dat[,Blueprint := stri_replace_all_fixed(Blueprint, "}", ' ')]
dat[,Blueprint := stri_replace_all_fixed(Blueprint, ",", ' ')]
dat[,Blueprint := stri_replace_all_fixed(Blueprint, ";", ' ')]
dat[,Blueprint := stri_replace_all_regex(Blueprint, " +", ' ')]

dat[,Blueprint := stri_trim_both(Blueprint)]

#Select some columns
dat <- dat[main_task != 'WNGER2',
  list(
    metric,
    Filename,
    metric,
    main_task,
    Sample_Pct,
    Total_Time_P1,
    Max_RAM_GB = Max_RAM / (1e+9),
    Gini.Norm_H,
    cv_method,
    blueprint_storage_MB = blueprint_storage_size_P1 / (1e+6),
    holdout_time_minutes = holdout_scoring_time / 60,
    model_training_time_hours = Total_Time_P1/3600,
    test,
    tasks=X_tasks,
    gs_metric
    )]

#Split problem types
dat[main_task == 'VWLRQC', main_task := 'VWLRQ']
dat[main_task == 'VWLRQR', main_task := 'VWLRQ']

dat[main_task == 'VWSPC', main_task := 'VWSP']
dat[main_task == 'VWSPR', main_task := 'VWSP']

dat[main_task == 'VWC', main_task := 'VW']
dat[main_task == 'VWR', main_task := 'VW']

#Sort files by max RAM
f <- dat[,list(run = max(Max_RAM_GB)), by='Filename']
f <- f[order(run, decreasing=TRUE),]
f_lev <- f$Filename
dat[,Filename := factor(Filename, levels=f_lev)]

#Sort main tasks by max RAM
t <- dat[,list(run = max(Max_RAM_GB)), by='main_task']
t <- t[order(run, decreasing=TRUE),]
t_lev <- t$main_task
dat[,main_task := factor(main_task, levels=t_lev)]

#Clean tasks
dat[,tasks := stri_replace_all_regex(tasks, "(SCTXT3/WNGER2/)+", 'SCTXT3/WNGER2/')]
dat[,tasks := stri_replace_all_regex(tasks, "BIND/", '/')]
dat[,tasks := stri_replace_all_regex(tasks, "/BIND", '/')]
dat[,tasks := stri_replace_all_regex(tasks, "//", '/')]
dat[,tasks := stri_replace_all_regex(tasks, "/$", '')]
dat[,tasks := stri_trim_both(tasks)]
dat[,task := stri_trim_both(tasks)]

#Make task
dat[, task := stri_paste(tasks, '/', main_task)]
dat[,task := stri_replace_all_regex(task, "^/", '')]
dat[,task := stri_replace_all_regex(task, "//", '/')]

#Reshape tall
id_vars <- c("Filename", "main_task", "Sample_Pct")
measure_vars <- c('Max_RAM_GB', 'model_training_time_hours', 'holdout_time_minutes', 'Gini.Norm_H')
dat <- melt.data.table(dat[,c(id_vars, measure_vars), with=FALSE], measure.vars=measure_vars)
dat[,variable := factor(variable, levels=measure_vars)]
dat <- dat[!is.na(value),]

#Reshape wide
dat <- dcast.data.table(dat, Filename + Sample_Pct + variable  ~ main_task)
dat[,lapply(.SD, function(x) sum(!is.na(x))), by='Filename']
dat[Filename=='Foursquare-daytime.csv',]
dat <- dat[!is.na(LENETCD) & ((!is.na(VWSP)) | (!is.na(VWLRQ)) | (!is.na(VW))),]

#color scale:
#http://colorbrewer2.org/
colors <- c(
  "#1f78b4", "#ff7f00", "#6a3d9a", "#33a02c", "#e31a1c", "#b15928",
  "#a6cee3", "#fdbf6f", "#cab2d6", "#b2df8a", "#fb9a99", "#ffff99", "black",
  rep("grey", 1000)
)
length(unique(colors))

#Plot overall stats
ggplot(dat, aes(x=LENETCD, y=VW, col=Filename)) +
  geom_point() +
  geom_abline(slope=1, intercept=0,linetype="dashed") +
  theme_bw() +
  scale_colour_manual(values=colors) +
  theme(legend.position = "bottom") +
  facet_wrap(~variable, scales='free', ncol=2)  +
  guides(col = guide_legend(nrow = 5, byrow = TRUE))


pdf('~/Documents/VW_vs_ENET.pdf', width=11, heigh=8.5)
#Plot Frozen stats - VW
ggplot(dat[Sample_Pct == 95,], aes(x=LENETCD, y=VW, col=Filename)) +
  geom_point() +
  geom_abline(slope=1, intercept=0,linetype="dashed") +
  theme_bw() +
  scale_colour_manual(values=colors) +
  theme(legend.position = "bottom") +
  facet_wrap(~variable, scales='free', ncol=2)  +
  guides(col = guide_legend(nrow = 5, byrow = TRUE)) +
  ggtitle('VW vs Elastic Net @95% frozen runs')

#Plot Frozen stats - VWLRQ
ggplot(dat[Sample_Pct == 95,], aes(x=LENETCD, y=VWLRQ, col=Filename)) +
  geom_point() +
  geom_abline(slope=1, intercept=0,linetype="dashed") +
  theme_bw() +
  scale_colour_manual(values=colors) +
  theme(legend.position = "bottom") +
  facet_wrap(~variable, scales='free', ncol=2)  +
  guides(col = guide_legend(nrow = 5, byrow = TRUE)) +
  ggtitle('VW Low-Rank Quadratic vs Elastic Net @95% frozen runs')

#Plot Frozen stats - VWSP
ggplot(dat[Sample_Pct == 95,], aes(x=LENETCD, y=VWSP, col=Filename)) +
  geom_point() +
  geom_abline(slope=1, intercept=0,linetype="dashed") +
  theme_bw() +
  scale_colour_manual(values=colors) +
  theme(legend.position = "bottom") +
  facet_wrap(~variable, scales='free', ncol=2)  +
  guides(col = guide_legend(nrow = 5, byrow = TRUE)) +
  ggtitle('VW Stagewise Polynomial vs Elastic Net @95% frozen runs')

#Plot Frozen stats - VW vs VWLRQ
ggplot(dat[Sample_Pct == 95,], aes(x=VW, y=VWLRQ, col=Filename)) +
  geom_point() +
  geom_abline(slope=1, intercept=0,linetype="dashed") +
  theme_bw() +
  scale_colour_manual(values=colors) +
  theme(legend.position = "bottom") +
  facet_wrap(~variable, scales='free', ncol=2)  +
  guides(col = guide_legend(nrow = 5, byrow = TRUE)) +
  ggtitle('VW Low-Rank Quadratic vs VW @95% frozen runs')

#Plot Frozen stats - VW vs VWSP
ggplot(dat[Sample_Pct == 95,], aes(x=VW, y=VWSP, col=Filename)) +
  geom_point() +
  geom_abline(slope=1, intercept=0, linetype="dashed") +
  theme_bw() +
  scale_colour_manual(values=colors) +
  theme(legend.position = "bottom") +
  facet_wrap(~variable, scales='free', ncol=2)  +
  guides(col = guide_legend(nrow = 5, byrow = TRUE)) +
  ggtitle('VW Stagewise Polynomial vs VW @95% frozen runs')
dev.off()

system('open ~/Documents/VW_vs_ENET.pdf')