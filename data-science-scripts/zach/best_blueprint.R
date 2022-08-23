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
#suppressWarnings(mbp_10_0_08 <- getLeaderboard('577fc68182785c15b465dba7'))
suppressWarnings(mbp_10_0_09 <- getLeaderboard('578f82e93df1565ececddb84')) # 578354f21cd43463746b1cfd
#suppressWarnings(mbp_10_0_10 <- getLeaderboard('579991a7c2e79c2d3dbcaa9c'))

#Subset
dat <- mbp_10_0_09
dat <- dat[Sample_Pct==64,]

#Check holdout size
dat[,holdout_pct := round(holdout_size/(Sample_Size/(Sample_Pct/100)),3)]
summary(dat$holdout_pct); dim(dat)

#Remove blenders and prime models
dat <- dat[(!is_blender),]
dat <- dat[(!is_prime),]
dat <- dat[main_task != 'BLENDER w OSSTASK',]
dat <- dat[main_task != 'TWSL2',]

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
setnames(dat, make.names(names(dat)))
dat <- dat[,
           list(
             Blueprint,
             Filename,
             main_task,
             sub_tasks=X_tasks,
             Sample_Pct,
             Gini.Norm_H,
             Max_RAM_GB = Max_RAM / (1e+9),
             blueprint_storage_MB = blueprint_storage_size_P1 / (1e+6),
             holdout_time_minutes = holdout_scoring_time / 60,
             model_training_time_minutes = Total_Time_P1 / 60
           )]

#Clean sub tasks
dat[,sub_tasks := stri_replace_all_regex(sub_tasks, "(SCTXT3/WNGER2/)+", 'SCTXT3/WNGER2/')]
dat[,sub_tasks := stri_replace_all_regex(sub_tasks, "(SCTXT3/WNGEC2/)+", 'SCTXT3/WNGEC2/')]
dat[,sub_tasks := stri_replace_all_regex(sub_tasks, "BIND/", '/')]
dat[,sub_tasks := stri_replace_all_regex(sub_tasks, "/BIND", '/')]
dat[,sub_tasks := stri_replace_all_regex(sub_tasks, "//", '/')]
dat[,sub_tasks := stri_replace_all_regex(sub_tasks, "/$", '')]
dat[,sub_tasks := stri_trim_both(sub_tasks)]
dat[,sub_tasks := stri_trim_both(sub_tasks)]
dat[,sort(table(sub_tasks))]

#Clean main tasks
dat[main_task %in% c('ENETCD', 'LENETCD', 'BENETCD', 'BLENETCD', 'ENETCDWC', 'LENETCDWC', 'LRCD', 'GLM', 'LR', 'GLMCD', 'LR1') , main_task := 'Linear/Elastic Models']
dat[main_task %in% c('RFR', 'RFC') , main_task := 'Random Forest']
dat[main_task %in% c('ESXGBR', 'ESXGBC', 'XGBR', 'XGBC', 'PXGBC', 'PXGBR') , main_task := 'XGBoost']
dat[main_task %in% c('ESGBC', 'ESGBR', 'GBR', 'GBC', 'PGBC', 'PGBR', 'MNGBR') , main_task := 'GBM']
dat[main_task %in% c('ETC', 'ETR') , main_task := 'Extra Trees']
dat[main_task %in% c('ASVMSKR', 'ASVMSKC', 'SVMR', 'SVMC', 'ASVMER', 'ASVMEC', 'ASVMR', 'ASVMC') , main_task := 'SVM']
dat[main_task %in% c('WNGEC2', 'WNGER2', 'CNGEC2') , main_task := 'Text Mining']
dat[main_task %in% c('RULEFITR', 'RULEFITC') , main_task := 'RuleFit']
dat[main_task %in% c('DTR', 'DTC') , main_task := 'Decision Tree']
dat[main_task %in% c('KNNC', 'KNNR') , main_task := 'KNN']
dat[main_task %in% c('SGDR', 'SGDC', 'SGDRA') , main_task := 'SGD']
dat[main_task %in% c('RR', 'RC') , main_task := 'Dummy Models']
dat[main_task %in% c('TWSL2') , main_task := '2-stage model']
dat[main_task %in% c('CNBC') , main_task := 'Naive Bayes']
dat[,list(.N), by='main_task'][order(N),]

#Remove
dat[,Blueprint := NULL]
dat <- dat[main_task != 'SGD',]

#Add overall metrics:
dat[,model_rank := rank(1-Gini.Norm_H, ties.method='min'), by=c('Filename', 'Sample_Pct')]
dat[,best_gini := max(Gini.Norm_H), by='Filename']
dat[,gini_diff := best_gini - Gini.Norm_H]

#Save data
write.csv(dat, '~/Documents/model_ranks.csv', row.names=FALSE)

#Summarize ranks
all <- dat[,list(.N), by='main_task'][order(N),]
all[,N := NULL]
out <- dat[model_rank==1, list(.N), by='main_task'][,list(main_task, pct_rank_1 = N/sum(N))][order(pct_rank_1, decreasing=TRUE),]
out <- merge(all, out, by='main_task', all.x=TRUE)
out <- out[order(pct_rank_1, decreasing=TRUE),]
out[is.na(pct_rank_1), pct_rank_1 := 0]
out[,list(main_task, pct_rank_1=round(pct_rank_1 * 100, 1))]
write.csv(out, row.names=FALSE, file='~/Documents/ranks.csv')

#Sort models by average rank
f <- dat[,list(rank = as.integer(median(model_rank)), rank2 = mean(model_rank)), by='main_task']
f <- f[order(rank, rank2),]
f_lev <- f$main_task
dat[,main_task := factor(main_task, levels=f_lev)]
stopifnot(levels(dat$main_task) == f_lev)

#color scale:
#http://colorbrewer2.org/
colors <- c(
  "#1f78b4", "#ff7f00", "#6a3d9a", "#33a02c", "#e31a1c", "#b15928",
  "#a6cee3", "#fdbf6f", "#cab2d6", "#b2df8a", "#fb9a99", "grey", "black", "#ffff99",
  rep("grey", 1000)
)

#Plot histograms
ggplot(dat, aes(x=model_rank, fill=main_task)) +
  geom_histogram() +
  theme_bw() +
  scale_fill_manual(values=colors) +
  theme(legend.position = "bottom") +
  facet_wrap(~main_task, scales='free', ncol=2)  +
  guides(col = guide_legend(nrow = 5, byrow = TRUE))

ggplot(dat, aes(x=gini_diff, fill=main_task)) +
  geom_histogram() +
  theme_bw() +
  scale_fill_manual(values=colors) +
  theme(legend.position = "bottom") +
  facet_wrap(~main_task, scales='free', ncol=2)  +
  guides(col = guide_legend(nrow = 5, byrow = TRUE))

#Violins
ggplot(dat, aes(x=main_task, y=model_rank, color=main_task)) +
  geom_violin(draw_quantiles = c(0.50)) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_color_manual(values=colors) +
  theme(legend.position = "bottom") +
  guides(col = guide_legend(nrow = 2, byrow = TRUE))
