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
# 573b6d60fc8c77689dd99b21 - MB 10_0_05
# 573dcc2f969e733da97c8aa4 - MB 10_0_04 baseline
# 5743a4f927f1753e1a972806 - XGB only 0.05
# 5743a41e6b342f3deacdc44d - XGB only 0.15
# 5743a4f927f1753e1a972806 - XGB only 0.30

#Load data
suppressWarnings(old_XGB <- loadLeaderboard('5743a4f927f1753e1a972806'))
suppressWarnings(new_XGB <- loadLeaderboard('573b6d60fc8c77689dd99b21'))
old_XGB[,test := 't0.05']
new_XGB[,test := 't0.30']

#Combine data
dat <- rbind(old_XGB, new_XGB, fill=TRUE, use.names=TRUE)
setnames(dat, make.names(names(dat), unique=TRUE))
dat <- dat[main_task %in% c('ESXGBC', 'ESXGBR'),]

#Clean some columns
setnames(dat, make.names(names(dat)))
dat[,Blueprint := sapply(Blueprint, function(x) paste(unlist(x), collapse=" "))]

#Extract learning rate / trees / cbt
dat[,lr := as.numeric(stri_match_first_regex(model_name, 'lr=(\\d.\\d+)')[,2])]
dat[,n := as.numeric(stri_match_first_regex(model_name, 'n=(\\d.\\d+)')[,2])]
dat[,lT := as.numeric(stri_match_first_regex(model_name, 'lT=(\\d.\\d+)')[,2])]
dat[,lTr := as.numeric(stri_match_first_regex(model_name, 'lTr=(\\d.\\d+)')[,2])]
dat[,cbt := stri_match_first_regex(model_name, 'cbt=(\\[.*?\\])')[,2]]
dat[,l := stri_match_first_regex(model_name, 'l=(.*?) ')[,2]]

#Fix model name
dat[,model_name := stri_replace_all_regex(model_name, 'wa_fp=[\\p{L}|\\p{N}]+', '')]
dat[,model_name := stri_replace_all_regex(model_name, 'lr=0\\.[\\p{N}]+', '')]
dat[,model_name := stri_replace_all_fixed(model_name, 'character(0)', '')]
dat[,model_name := stri_replace_all_fixed(model_name, 'ESXGBR', '')]
dat[,model_name := stri_replace_all_fixed(model_name, 'ESXGBC', '')]
dat[,model_name := stri_trim(model_name)]
#dat[,table(model_name)]

#Fix tasks
dat[, task := paste(main_task, "-", X_tasks)]
dat[, task := gsub("/BIND", "", task, fixed=TRUE)]
dat[, task := gsub("SCTXT2/WNGER2/", "wnger2", task, fixed=TRUE)]
dat[, task := gsub("SCTXT2/CNGEC2/", "cngec2", task, fixed=TRUE)]
dat[, task := gsub("SCTXT2/CNGER2/", "cnger2", task, fixed=TRUE)]
dat[, task := gsub("(wnger2)+", "WNGER2/", task, fixed=FALSE)]
dat[, task := gsub("(cngec2)+", "CNGEC2/", task, fixed=FALSE)]
dat[, task := gsub("(cnger2)+", "CNGER2/", task, fixed=FALSE)]
#dat[,table(task)]

#Select some columns
dat <- dat[,
  list(
    metric,
    Filename,
    metric,
    main_task,
    task,
    model_name,
    Blueprint,
    Sample_Pct,
    test,
    Total_Time_P1,
    Max_RAM_GB,
    Gini.Norm_H,
    max_vertex_storage_size_P1,
    blueprint_storage_size_P1,
    cv_method,
    Max_RAM_MB,
    Max_RAM_GB,
    blueprint_storage_MB,
    max_vertex_size_MB,
    holdout_time_seconds,
    holdout_time_minutes,
    model_training_time_minutes
    )]

#Sort files by max RAM
f <- dat[,list(run = max(Max_RAM_GB)), by='Filename']
f <- f[order(run, decreasing=TRUE),]
f_lev <- f$Filename
#dat[,Filename := factor(Filename, levels=f_lev)]

#Sort main tasks by max RAM
t <- dat[,list(run = max(Max_RAM_GB)), by='main_task']
t <- t[order(run, decreasing=TRUE),]
t_lev <- t$main_task
#dat[,main_task := factor(main_task, levels=t_lev)]

#Reshape tall
id_vars <- c("Filename", "Blueprint", "main_task", "task", "model_name", "Sample_Pct", "test")
measure_vars <- c('Max_RAM_GB', 'model_training_time_minutes', 'holdout_time_minutes', 'Gini.Norm_H', "blueprint_storage_MB", "max_vertex_size_MB")
dat <- melt.data.table(dat[,c(id_vars, measure_vars), with=FALSE], measure.vars=measure_vars)
dat[,variable := factor(variable, levels=measure_vars)]
dat <- dat[!is.na(value),]

#Reshape wide
dat <- dcast.data.table(dat, Filename + main_task + task + model_name + Sample_Pct + variable ~ test)
dat <- dat[!is.na(t0.05),]
dat <- dat[!is.na(t0.30),]
dat
dat[,table(Filename, t0.05)]
dat[,table(Filename, t0.30)]

#color scale:
#http://colorbrewer2.org/
colors <- c(
  "#1f78b4", "#ff7f00", "#6a3d9a", "#33a02c", "#e31a1c", "#b15928",
  "#a6cee3", "#fdbf6f", "#cab2d6", "#b2df8a", "#fb9a99", "#ffff99", "black",
  "grey", "grey"
)
colors[duplicated(colors)]
length(unique(colors))

#Plot overall stats
ggplot(dat[Sample_Pct==95,], aes(x=value, y=main_task, col=Filename, label=Filename)) +
  geom_point() +
  theme_bw() +
  scale_colour_manual(values=colors) +
  theme(legend.position = "bottom") +
  facet_wrap(~variable, scales='free', ncol=2)  +
  theme(axis.text.y = element_text(size=6)) +
  guides(col = guide_legend(nrow = 5, byrow = TRUE))

#Detailed report
f <- '~/detailed_10GB_report.pdf'
unlink(f)
pdf(f, width=11, height=8.5)
for(var in measure_vars){
  p1 <- ggplot(dat[variable == var,], aes(x=value, y=task, col=Filename, label=Filename)) +
    geom_point() +
    theme_bw() +
    scale_colour_manual(values=colors) +
    theme(legend.position = "bottom") +
    facet_wrap(~variable, scales='free', ncol=1) +
    theme(axis.text.y = element_text(size=6)) +
    guides(col = guide_legend(nrow = 4, byrow = TRUE)) +
    theme(legend.text=element_text(size=8))
  print(p1)
}
dev.off()
system(paste('open', f))
