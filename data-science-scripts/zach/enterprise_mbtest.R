#Libraries
library(data.table)
library(ggplot2)
library(reshape2)
library(scales)

#Load Data
dat_full <- read.csv(unz("~/workspace/data-science-scripts/data/lb_export.csv.zip", "lb_export.csv"))
dat_full <- data.table(dat_full)
#sort(unique(dat$Filename))
sort(names(dat_full))

#4 sets
dat <- dat_full[
  Filename %in% c(
    'mini_boone_full.csv',
    'mini_boone_small.csv',
    '28_Features_split_train_converted.csv',
    'grockit_train_small_no_outcome.csv'),
  list(
    category, is_blender, is_prime, quickrun_model, reference_model, Total_Time_P1,
    main_task, Sample_Pct, Filename, max_vertex_task_name_P1,
    Blueprint)
  ]
dat[category == '', category := 'NewMBTest']
dat <- dcast.data.table(
  dat, Filename + Blueprint + Sample_Pct + quickrun_model + reference_model + is_blender + is_prime + main_task + max_vertex_task_name_P1 ~ category,
  value.var='Total_Time_P1')
dat[,Blueprint := NULL]

ggplot(dat, aes(x=Benchmark, y=NewMBTest)) +
  geom_point() +
  geom_abline(slope=1, intercept=0) +
  geom_text(aes(label=main_task), size=3, vjust=0, hjust=-.1, angle = -90) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  coord_fixed() +
  expand_limits(x = 0, y = 0) +
  facet_wrap(~Filename, scales='free')

#ESGBC only
dat <- dat_full[
  main_task == 'ESGBC',
  list(
    category, main_task, Sample_Pct, Filename,
    max_vertex_task_name_P1,
    is_blender, is_prime, quickrun_model, reference_model,
    Total_Time_P1,
    holdout_scoring_time,
    blueprint_storage_size_P1,
    Max_RAM,
    Blueprint)
  ]
levs <- dat[,list(t=max(Total_Time_P1)),by='Filename']
levs <- levs[order(t),as.character(Filename)]
dat[,Filename := factor(Filename, levels=levs)]
dat[category == '', category := 'NewMBTest']
dat[,Max_RAM := Max_RAM * 1e-9]
dat[,blueprint_storage_size_P1 := blueprint_storage_size_P1 * 1e-9]

dat <- melt(dat, measure.vars=c(
  'Total_Time_P1', 'holdout_scoring_time',
  'blueprint_storage_size_P1', 'Max_RAM'))

dat <- dcast.data.table(
  dat, Filename + Blueprint + Sample_Pct +
    quickrun_model + reference_model +
    is_blender + is_prime + main_task +
    max_vertex_task_name_P1 +
    variable ~ category)
head(dat)

ggplot(dat, aes(x=Benchmark, y=NewMBTest)) + geom_point() +
  geom_abline(slope=1, intercept=0) +
  coord_fixed() +
  expand_limits(x = 0, y = 0) +
  theme_bw() +
  scale_x_log10(labels = comma) +
  scale_y_log10(labels = comma) +
  facet_wrap(~variable, scales='free')



###OLD
ggplot(dat, aes(x=Filename, y=holdout_scoring_time, col=category)) +
  geom_point() + coord_flip() + theme_bw() +
  theme(legend.position="top") +
  scale_color_brewer(type='qual', palette=1) +
  scale_y_log10() +
  ggtitle('Enterprise MBtest ESGBC vs Benchmark holdout_scoring_time')



