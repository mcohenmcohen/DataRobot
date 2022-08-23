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

#MBtest IDs
# 577f0a52d241bf70182ba069 master
# 5797b01c196da75f42adc7b5 S3 cluster

#Load data
suppressWarnings(Base <- getLeaderboard('577f0a52d241bf70182ba069'))
suppressWarnings(S3 <- getLeaderboard('5797b01c196da75f42adc7b5'))
Base[,test := 'Shrink']
S3[,test := 'S3']

#Combine data
dat <- rbind(Base, S3, fill=TRUE, use.names=TRUE)
setnames(dat, make.names(names(dat), unique=TRUE))
setnames(dat, make.names(names(dat), unique = TRUE))

#Subset
#dat[,table(Sample_Pct, test)]
#dat <- dat[Sample_Pct==95,]

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
dat[,Blueprint := stri_replace_all_fixed(Blueprint, "dtype=float32", ' ')]
dat[,Blueprint := stri_replace_all_fixed(Blueprint, "dtype=float64", ' ')]
dat[,Blueprint := stri_replace_all_fixed(Blueprint, "rs=1234", ' ')]

dat[,Blueprint := stri_replace_all_regex(Blueprint, " [1-9]+ ", ' ')]
dat[,Blueprint := stri_replace_all_regex(Blueprint, "cn=[(a-z)|(0-9)]+", ' ')]
dat[,Blueprint := stri_replace_all_regex(Blueprint, " +", ' ')]

dat[,Blueprint := stri_trim_both(Blueprint)]

#Select some columns
dat <- dat[main_task != 'WNGER2',
  list(
    metric,
    Filename,
    metric,
    main_task,
    Blueprint,
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
    tasks_args,
    reference_model,
    quickrun_model,
    is_blender,
    main_featsel,
    other_featsel,
    Task_Info_Extras,
    is_prime,
    main_args,
    stacked_args
    )]

#Uniques
dim(dat)
dat <- unique(dat)
dim(dat)

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
id_vars <- c("Filename", "main_task", "Blueprint", "Sample_Pct", "test")
measure_vars <- c('Max_RAM_GB', 'model_training_time_hours', 'holdout_time_minutes', 'Gini.Norm_H')
dat <- melt.data.table(dat[,c(id_vars, measure_vars), with=FALSE], measure.vars=measure_vars)
dat[,variable := factor(variable, levels=measure_vars)]
dat <- dat[!is.na(value),]

#Reshape wide
dat <- dcast.data.table(dat, Filename + main_task + Blueprint + Sample_Pct + variable ~ test)
dat <- dat[!is.na(Shrink) & !is.na(S3),]

#Debug
dat[S3 > 1 | Shrink > 1, table(as.character(main_task))]
dat[,summary(Sample_Pct)]

#color scale:
#http://colorbrewer2.org/
colors <- c(
  "#1f78b4", "#ff7f00", "#6a3d9a", "#33a02c", "#e31a1c", "#b15928",
  "#a6cee3", "#fdbf6f", "#cab2d6", "#b2df8a", "#fb9a99", "#ffff99", "black",
  rep("grey", 1000)
)
length(unique(colors))

#Plot overall stats
ggplot(dat, aes(x=Shrink, y=S3, col=Filename)) +
  geom_point() +
  geom_abline(slope=1, intercept=0,linetype="dashed") +
  theme_bw() +
  scale_colour_manual(values=colors) +
  theme(legend.position = "bottom") +
  facet_wrap(~variable, scales='free', ncol=2)  +
  guides(col = guide_legend(nrow = 5, byrow = TRUE))
