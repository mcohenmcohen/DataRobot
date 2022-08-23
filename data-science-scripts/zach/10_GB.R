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
# 571fc2209db768539aa7d280 - 1st run, no frozen
# 5722578ad5ccae1473b9c49e - 2nd run, frozen more datasets
# 5726810e4721e514f99b7936 - 3rd run frozen
# 5727bcf68e8588169b6def50 - MB 10_0_05
# 573113d5261fa8766bbff712 - ES XGB tweak
# 573a04fe6b5d473f52186a27 - MB 10_0_04 95% frozen
# 573a0c1f8426a542513336eb - MB 10_0_04 95% frozen - new XGB lr
# 573b6d60fc8c77689dd99b21 - MB 10_0_05
# 573dcc2f969e733da97c8aa4 - MB 10_0_04 baseline
# 5743a4f927f1753e1a972806 - XGB only 0.05
# 5743a41e6b342f3deacdc44d - XGB only 0.15
# 5743a4f927f1753e1a972806 - XGB only 0.30
# 57743297a253e67485b49671 - 10_0_08 master

#Load base MBtest and new run
#Sys.sleep(43200)
suppressWarnings(dat <- getLeaderboard('5797b01c196da75f42adc7b5'))
setnames(dat, make.names(names(dat), unique=TRUE))
dat[,sort(table(main_task))]
dat[,sort(table(Filename))]
#dat[,table(Filename, Sample_Pct, quickrun_model)]

#Show max sample size
dat[,list(samp=max(Sample_Pct)), by='Filename']
dat[,list(samp=max(Sample_Pct)), by='main_task'][order(samp),]

#Subset data
length(unique(dat$Filename))
setnames(dat, make.names(names(dat)))
dat[,Blueprint := lapply(Blueprint, toJSON, auto_unbox = T, pretty=T)]
dat[,Task_Info_Extras := lapply(Task_Info_Extras, toJSON, auto_unbox = T, pretty=T)]

#Fix model name
dat[,model_name := stri_replace_all_regex(main_task, 'wa_fp=[\\p{L}|\\p{N}]+', '')]
dat[,model_name := stri_replace_all_fixed(model_name, 'l=tweedie', '')]
dat[,model_name := stri_replace_all_fixed(model_name, 'l=poisson', '')]
dat[,model_name := stri_replace_all_fixed(model_name, 'character(0)', '')]
dat[,model_name := stri_replace_all_fixed(model_name, 'ESXGBR', '')]
dat[,model_name := stri_replace_all_fixed(model_name, 'ESXGBC', '')]
dat[,model_name := stri_trim(model_name)]
#dat[,table(model_name)]

dat[Sample_Pct == 95, sort(unique(Sample_Size))]

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
    Max_RAM_GB = as.numeric(Max_RAM/1024^3),
    Gini.Norm_H,
    RMSE_H,
    max_vertex_storage_size_P1,
    blueprint_storage_size_P1,
    X_tasks,
    Task_Info_Extras,
    cv_method,
    blueprint_storage_MB = blueprint_storage_size_P1/(1024^2),
    max_vertex_size_MB = max_vertex_storage_size_P1/(1024^2),
    holdout_time_seconds = holdout_scoring_time/1000,
    holdout_time_minutes = holdout_scoring_time/60000,
    model_training_time_minutes = Total_Time_P1/60
    )]

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

#Write sample pct vs runtime vs ram
dat[,Blueprint := as.character(lapply(Blueprint, paste, collapse=' '))]
dat[,sample_number := NULL]
dat[, sample_number := as.integer(factor(Sample_Pct)), by=c('Filename', 'Blueprint')]
#dat[,table(main_task, sample_number)]
dat[,table(sample_number)]
dat[,sample_number := paste0('autopilot_', sample_number)]

#Reshape
#Can also do RMSE_H
id_vars <- c("Filename", "main_task", "cv_method", "metric", "X_tasks", "Blueprint", "Sample_Pct", "sample_number")
measure_vars <- c('Max_RAM_GB', 'model_training_time_minutes', 'holdout_time_minutes', 'Gini.Norm_H', "blueprint_storage_MB", "max_vertex_size_MB")
dat <- melt.data.table(dat[,c(id_vars, measure_vars), with=FALSE], measure.vars=measure_vars)
dat[,variable := factor(variable, levels=measure_vars)]
dat <- dat[!is.na(value),]
sort(sapply(dat, function(x) length(unique(x))))

#color scale:
#http://colorbrewer2.org/
colors <- c(
  "#1f78b4", "#ff7f00", "#6a3d9a", "#33a02c", "#e31a1c", "#b15928",
  "#a6cee3", "#fdbf6f", "#cab2d6", "#b2df8a", "#fb9a99", "#ffff99", "black",
  rep("grey", 100)
)
colors[duplicated(colors)]
length(unique(colors))

#Plot overall stats
plt <- dat
plt[variable == 'model_training_time_minutes', max(value) / 60]
ggplot(plt, aes(x=value, y=main_task, col=Filename, label=Filename)) +
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
  p1 <- ggplot(dat[variable == var,], aes(x=value, y=main_task, col=Filename, label=Filename)) +
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
