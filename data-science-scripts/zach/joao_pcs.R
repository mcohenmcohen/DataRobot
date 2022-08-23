#Load Data
library('shrinkR')
library('data.table')
library('reshape2')
library('ggplot2')
library('ggrepel')
library('scales')
library('jsonlite')

#MBtest IDs
# 5717bb91c688893bdf89dd72 - 1 dataset, successful run
# 571a37c28b186a52fcd7e450 - multi dataset
# 571a667d9941c1062ec345eb - multi dataset
# 571fb5327e1bd20fd73c8816 - multi dataset post ram fix
# 57221d478b6eea1407a4f458 - multi dataset post ram fix 2
# 57239d5424ac7b140dc20131 - multi dataset, no stops + tfidf
# 57366346ade8a83e54e9e1e4 - multi dataset, no stops + tfidf

#check <- verifyYaml('~/workspace/datarobot/tests/ModelingMachine/out_of_core_rulefit_datasets_largedata_regonly.yaml')

#Load base MBtest and new run
#Sys.sleep(43200)
dat <- loadLeaderboard('57366346ade8a83e54e9e1e4')
setnames(dat, make.names(names(dat), unique=TRUE))
dat[,sort(table(main_task))]
dat[,sort(table(Filename))]

#Show max sample size
dat[,list(samp=max(Sample_Pct)), by='Filename']

#Pull out PTM2 task
dat[,PTM2 := ""]
dat[,PTM2 := sapply(Blueprint, function(x){
  a <- unlist(x)
  a <- a[grepl('^PTM2 ', a)]
  a[1]
})]
dat[is.na(PTM2), PTM2 := ""]

#Subset data
length(unique(dat$Filename))
setnames(dat, make.names(names(dat)))
dat[,Blueprint := lapply(Blueprint, toJSON, auto_unbox = T, pretty=T)]
dat[,Task_Info_Extras := lapply(Task_Info_Extras, toJSON, auto_unbox = T, pretty=T)]

dat <- dat[
  main_task %in% c(
    "ENETCD", "ESXGBR",
    "LENETCD", "ESXGBC"
    ),
  list(
    metric,
    Filename,
    is_blender,
    metric,
    main_task,
    Blueprint,
    Sample_Pct,
    cv_method,
    Total_Time_P1,
    Max_RAM_GB,
    holdout_scoring_time,
    Gini.Norm_H,
    RMSE_H,
    X_tasks,
    Task_Info_Extras,
    cv_method,
    PTM2)]
dat[, cosine := ifelse(grepl('pcs', tolower(X_tasks)), 'cosine', 'no cosine')]
dat[main_task %in% c("ENETCD", "LENETCD"), main_task := 'ENET']
dat[main_task %in% c("ESXGBR", "ESXGBC"), main_task := 'XGB']

#MOVE OVER TO shrinkR package
dat[,holdout_time_seconds := holdout_scoring_time/1000]
dat[,holdout_scoring_time := NULL]

#Say that anti-predictive == predictive
#(or we could cap)
dat[, Gini.Norm_H := abs(Gini.Norm_H)]

#Sort files by max runtime
f <- dat[,list(run = max(Total_Time_P1)), by='Filename']
f <- f[order(run),]
f_lev <- f$Filename

#Reshape
#Can also do RMSE_H
dat[,model := paste(main_task, PTM2)]
dat[,model := stri_replace_all_regex(model, " +", " ")]
dat[,model := stri_replace_all_fixed(model, ";spf=0", "")]
dat[,model := stri_trim(model)]

id_vars <- c("Filename", "Blueprint", "Sample_Pct", "cv_method", "metric", "cosine", "model", "main_task")
measure_vars <- c('Total_Time_P1', 'holdout_time_seconds', 'Gini.Norm_H', 'Max_RAM_GB')
x <- melt.data.table(dat[,c(id_vars, measure_vars), with=FALSE], measure.vars=measure_vars)
x <- x[!is.na(value),]
x <- dcast.data.table(x, Filename + Sample_Pct + cv_method + main_task + model + cosine ~ variable)
x[,Filename := factor(Filename, levels=f_lev)]
x

#Plot - 2 way
ggplot(x, aes(x=Gini.Norm_H, y=Filename, col=model, label=Filename)) +
  geom_point() +
  #geom_label_repel() + # Turn on for multiple datasets
  theme_bw() +
  scale_colour_manual(values=c('#984ea3', '#4daf4a', '#377eb8', '#ff7f00', '#e41a1c')) +
  theme(legend.position = "bottom") +
  expand_limits(x = 0, y = 0) +
  facet_grid(main_task ~ cosine) +
  guides(colour = guide_legend(nrow=2))

