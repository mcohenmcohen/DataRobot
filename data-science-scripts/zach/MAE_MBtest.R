library(data.table)
library(reshape2)
library(ggplot2)
library(matrixStats)

#Test 1 MAE only (some errors) 589a3965f66028108fb39b60
#Test 1 full 200 only (some errors) 589ce0084e0a9b1062fc85db

#Test 2 MAE only (no errors) 589de4389a2c824938b63d50
#Test 2 full 200 only (one error due to bad target) 589e1a168ccd292da3b305aa

#Test 3 full 200 only (no errors) 58a62767562adc29b0c546b9

#Test 4 full 200 only, 200 vs 100 min (no errors) 58a85d2ccf4c8711d7bc5e78

#Test 5 83 MAE only, (no errors) 58b492dd8fac2b1094d5554d
#Test 6 200, only 6 finished due to budget

dat_full <- fread('http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests=58b492dd8fac2b1094d5554d&max_sample_size_only=false')

dat <- dat_full[Sample_Pct == 80 & is_blender == TRUE,]
dat[,main_task := gsub('BLENDER w ', '', main_task)]
dat[,main_task := gsub('BL$', '', main_task)]
dat[grepl('GLM', main_task), main_task := 'GLM']
dat[grepl('ENET', main_task), main_task := 'ENET']
dat[grepl('l1_penalty', Blueprint), main_task := 'MAEL1']
dat[,sort(table(main_task))]
setnames(dat, make.names(names(dat), unique=TRUE))
dat <- dat[,list(
  Filename,
  main_task,
  Max_RAM,
  Total_Time_P1,
  error_H,
  MAD_H,
  holdout_size,
  holdout_scoring_time
)]
dat[,Max_RAM := Max_RAM / (1024 * 1024)]
setnames(dat, 'Max_RAM', 'Max_RAM_MB')

dat[,holdout_scoring_time := holdout_scoring_time / 1000]
setnames(dat, 'holdout_scoring_time', 'holdout_scoring_time_sec')

dat <- melt.data.table(dat, id.vars=c('Filename', 'main_task'))
dat <- dcast.data.table(dat, Filename + variable ~ main_task)
dat <- dat[variable %in% c('Max_RAM_MB', 'Total_Time_P1', 'MAD_H', 'holdout_scoring_time_sec'),]
#dat[variable == 'MAD_H', value := log1p(value)]

dat <- dat[Filename != 'rollingsales_brooklyn_80.csv',]

#MAE vs GLM
ggplot(dat[!is.na(MAE),], aes(x=GLM, y=MAE)) +
  geom_point() + geom_abline(slope=1, intercept=0) +
  facet_wrap(~variable, scales='free') +
  theme_bw() + scale_x_log10() + scale_y_log10()
round(dat[variable == 'MAD_H', summary((GLM - MAE) / GLM)], 3) * 100

#MAEL1 vs ENET
ggplot(dat[!is.na(MAEL1),], aes(x=ENET, y=MAEL1)) +
  geom_point() + geom_abline(slope=1, intercept=0) +
  facet_wrap(~variable, scales='free') +
  theme_bw() + scale_x_log10() + scale_y_log10()
round(dat[variable == 'MAD_H', summary((ENET - MAEL1) / ENET)], 3) * 100

#Best new vs best old
dat[,best_old := rowMins(cbind(AVG, MED, GLM, ENET), na.rm=T)]
dat[,best_new := rowMins(cbind(MAE, MAEL1), na.rm=T)]
ggplot(dat, aes(x=best_old, y=best_new)) +
  geom_point() + geom_abline(slope=1, intercept=0) +
  facet_wrap(~variable, scales='free') +
  theme_bw() + scale_x_log10() + scale_y_log10()
round(dat[variable == 'MAD_H', summary((best_old - best_new) / best_old)], 3) * 100
dat[variable == 'MAD_H', summary((best_old - best_new))]
