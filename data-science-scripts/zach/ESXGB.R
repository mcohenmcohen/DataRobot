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

if(FALSE){
  info_10GB <- validateYaml("~/workspace/mbtest-datasets/mbtest_datasets/data/Large/MBtest_in_mem_10GB.yaml")
  print(info_10GB)
  info_5GB <- validateYaml("~/workspace/mbtest-datasets/mbtest_datasets/data/Large/MBtest_in_mem_5GB.yaml")
  print(info_5GB)
}

#MBtest IDs
# 572cbc8df030fa588743f6a8 - 1st run
# 573113d5261fa8766bbff712 - marius' new heuristics
# 57360003a9189c75bcad2509 - my new heuristics

#Load base MBtest and new run
#Sys.sleep(43200)
suppressWarnings(dat <- loadLeaderboard('57360003a9189c75bcad2509'))
setnames(dat, make.names(names(dat), unique=TRUE))
dat[,sort(table(main_task))]
dat[,sort(table(task))]
dat[,sort(table(Filename))]
#dat[,table(Filename, Sample_Pct, quickrun_model)]

#Subset data
dat <- dat[main_task %in% c('ESXGBC', 'ESXGBR'),]
length(unique(dat$Filename))
setnames(dat, make.names(names(dat)))
dat[,Blueprint := lapply(Blueprint, toJSON, auto_unbox = T, pretty=T)]
dat[,Task_Info_Extras := lapply(Task_Info_Extras, toJSON, auto_unbox = T, pretty=T)]

#Fix model name
dat[,model_name := stri_replace_all_regex(model_name, 'wa_fp=[\\p{L}|\\p{N}]+', '')]
dat[,model_name := stri_replace_all_fixed(model_name, 'l=tweedie', '')]
dat[,model_name := stri_replace_all_fixed(model_name, 'l=poisson', '')]
dat[,model_name := stri_replace_all_fixed(model_name, 'character(0)', '')]
dat[,model_name := stri_replace_all_fixed(model_name, 'ESXGBR', '')]
dat[,model_name := stri_replace_all_fixed(model_name, 'ESXGBC', '')]
dat[,model_name := stri_trim(model_name)]
#dat[,table(model_name)]

#Show max sample size
dat[,list(samp=max(Sample_Pct)), by='Filename']
#dat[,list(samp=max(Sample_Pct)), by='model_name'][order(samp),]

dat <- dat[,
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
    Gini.Norm_H,
    RMSE_H,
    max_vertex_storage_size_P1,
    blueprint_storage_size_P1,
    X_tasks,
    Task_Info_Extras,
    cv_method,
    Max_RAM_MB,
    Max_RAM_GB,
    blueprint_storage_MB,
    max_vertex_size_MB,
    holdout_time_seconds,
    holdout_time_minutes,
    model_training_time_minutes,
    task,
    model_name
    )]

#Extract learning rate / trees / cbt
dat[,lr := as.numeric(stri_match_first_regex(model_name, 'lr=(\\d.\\d+)')[,2])]
dat[,n := as.numeric(stri_match_first_regex(model_name, 'n=(\\d.\\d+)')[,2])]
dat[,lT := as.numeric(stri_match_first_regex(model_name, 'lT=(\\d.\\d+)')[,2])]
dat[,lTr := as.numeric(stri_match_first_regex(model_name, 'lTr=(\\d.\\d+)')[,2])]
dat[,cbt := stri_match_first_regex(model_name, 'cbt=(\\[.*?\\])')[,2]]

dat[is.na(lT), lT := 0]
dat[is.na(lTr), lTr := 0]
dat[, cbt := addNA(cbt)]

dat[,table(lr)]
dat[,table(n)]
dat[,table(lT)]
dat[,table(lTr)]
dat[,table(cbt)]

dat[, jolt := factor(lT != 0)]
dat[, experiment := paste(lr, n, ifelse(lT==0, '', 'jolt'))]

#Simple regression
dat[,Filename := factor(Filename)]
mod <- lm(Total_Time_P1 ~ n + lr + lT + lTr + cbt + jolt + Filename, dat)
round(coef(summary(mod)), 2)

#Say that anti-predictive == predictive
#(or we could cap)
dat[, Gini.Norm_H := abs(Gini.Norm_H)]

#Sort files by max RAM
f <- dat[,list(run = max(Max_RAM_GB)), by='Filename']
f <- f[order(run, decreasing=TRUE),]
f_lev <- f$Filename
print(f)
dat[,Filename := factor(Filename, levels=f_lev)]

#Sort main tasks by max RAM
t <- dat[,list(run = max(Max_RAM_GB)), by='main_task']
t <- t[order(run, decreasing=TRUE),]
t_lev <- t$main_task
dat[,main_task := factor(main_task, levels=t_lev)]

#Sort tasks by max RAM
#table(dat$model_name)
t <- dat[,list(run = max(Max_RAM_GB)), by='model_name']
t <- t[order(run, decreasing=TRUE),]
t_lev <- t$model_name
dat[,model_name := factor(model_name, levels=t_lev)]

#Write sample pct vs runtime vs ram
dat[,Blueprint := as.character(lapply(Blueprint, paste, collapse=' '))]
dat[,sample_number := NULL]
dat[, sample_number := as.integer(factor(Sample_Pct)), by=c('Filename', 'model_name')]
dat[,table(Sample_Pct, sample_number)]

#Reshape
#Can also do RMSE_H
id_vars <- c("Filename", "model_name", "Blueprint", "Sample_Pct", "sample_number", "experiment", "n", "lr", "jolt", "lT", 'lTr', 'cbt')
measure_vars <- c('Max_RAM_GB', 'model_training_time_minutes', 'Gini.Norm_H', "blueprint_storage_MB")
dat <- melt.data.table(dat[,c(id_vars, measure_vars), with=FALSE], measure.vars=measure_vars)
dat[,variable := factor(variable, levels=measure_vars)]
dat <- dat[!is.na(value),]
sort(sapply(dat, function(x) length(unique(x))))

#Table
x <- dat[,list(
  #min = min(value),
  #med = median(value)
  max = max(value)
), by=c('model_name', 'variable')]
x <- x[variable %in% c('Gini.Norm_H', 'model_training_time_minutes'),]
x <- melt.data.table(x, measure.vars=c('max'))
x <- dcast.data.table(x, model_name ~ variable + variable.1)
x[order(model_training_time_minutes_max, decreasing=T),]

#color scale:
#http://colorbrewer2.org/
colors <- c(
  "#1f78b4", "#ff7f00", "#6a3d9a", "#33a02c", "#e31a1c", "#b15928",
  "#a6cee3", "#fdbf6f", "#cab2d6", "#b2df8a", "#fb9a99", "#ffff99", "black",
  rep("grey", 50)
)
#colors[duplicated(colors)]
length(unique(colors))

#Plot overall stats
ggplot(dat, aes(x=value, y=model_name, col=experiment)) +
  geom_point() +
  theme_bw() +
  scale_colour_manual(values=colors) +
  theme(legend.position = "bottom") +
  facet_wrap(~variable, scales='free', ncol=2)  +
  theme(axis.text.y = element_text(size=6)) +
  guides(col = guide_legend(ncol = 5, byrow = TRUE))

plotdat <- dcast.data.table(
  dat, Filename + model_name + Sample_Pct + n + lr + lT + lTr +
    jolt + experiment + sample_number ~ variable)

#Plot overall stats
plt <- plotdat[Sample_Pct==100,]
plt[,model_name := factor(model_name)]
ggplot(plt, aes(x=model_training_time_minutes, y=Gini.Norm_H, col=experiment)) +
  geom_point() +
  theme_bw() +
  facet_wrap(~Filename, scales='free')  +
  scale_colour_manual(values=colors) +
  theme(legend.position = "bottom") +
  theme(axis.text.y = element_text(size=6))

#Plot autopilot stats
plt <- plotdat[sample_number==1,] #I think we just run max autopilot size
plt[,model_name := factor(model_name)]
ggplot(plt, aes(x=model_training_time_minutes, y=Gini.Norm_H, col=experiment)) +
  geom_point() +
  theme_bw() +
  facet_wrap(~Filename, scales='free')  +
  scale_colour_manual(values=colors) +
  theme(legend.position = "bottom") +
  theme(axis.text.y = element_text(size=6))
dev.off()

#Jolt vs baseline
pdf('~/Downloads/mbtesting.pdf', width=10.5, height=8.5)
plt <- plotdat[sample_number==1,]
plt[,best := 1:.N == which.max(Gini.Norm_H / model_training_time_minutes), by=c('Filename', 'jolt')]
plt <- plt[best == TRUE,]
plt[,MBP := ifelse(jolt=='TRUE', 'method2', 'method1')]
plt <- melt.data.table(
  plt,
  id.vars=c('MBP', 'Filename', 'Sample_Pct', 'experiment', 'model_name'),
  measure.vars=c('model_training_time_minutes', 'Gini.Norm_H')
  )
plt <- dcast.data.table(plt, Filename + Sample_Pct + variable ~ MBP)
plt
ggplot(plt, aes(x=method1, y=method2, col=Filename)) +
  geom_point() +
  theme_bw() +
  scale_colour_manual(values=colors) +
  facet_wrap(~variable, scales='free', ncol=1)  +
  theme(legend.position = "bottom") +
  geom_abline(intercept=0, slope=1, linetype = 2)
dev.off()
system('open ~/Downloads/mbtesting.pdf')
