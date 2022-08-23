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
library(reshape2)

# Custom color scale
colors <- c(
  "#1f78b4", "#ff7f00", "#6a3d9a", "#33a02c", "#e31a1c", "#b15928",
  "#a6cee3", "#fdbf6f", "#cab2d6", "#b2df8a", "#fb9a99", "black",
  "grey1", "grey10"
)

# Load Raw data
lb_4.0_raw <- fread('http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests=59e0b66c9d48cf0001cc782f&max_sample_size_only=false')
lb_3.1_raw <- fread('http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests=5968145f79c0b733c4c4fa93&max_sample_size_only=false')

# Combine the 2 datasets
lb_4.0 <- copy(lb_4.0_raw)
lb_3.1 <- copy(lb_3.1_raw)
lb_4.0[,MBP := 'mbp11_0_10']
lb_3.1[,MBP := 'mbp11_0_09']
dat <- rbind(lb_4.0, lb_3.1, fill=T)

# Format some units
dat[,Max_RAM_GB := as.numeric(Max_RAM) * 1e-9]
dat[,Total_Time_P1_Hours := Total_Time_P1/3600]
dat[,size_GB := size * 1e-9]
dat[,Sample_Pct := as.numeric(Sample_Pct)]
dat[,Filename := gsub('.csv', '', Filename, fixed=T)]
dat[,Filename := gsub('.sas7bdat', '', Filename, fixed=T)]
dat[,Filename := gsub('.zip', '', Filename, fixed=T)]

# Max ram by file / bp
dat <- dat[!is.na(Sample_Pct),]
dat <- dat[!is.na(Max_RAM_GB),]

# Plot max models
agg <- dat[,list(
  Max_RAM_GB=max(Max_RAM_GB, na.rm=T),
  worst_model=main_task[which.max(Max_RAM_GB)]
), by=list(Filename, MBP)]
agg <- melt.data.table(agg, measure.vars=c('Max_RAM_GB', 'worst_model'))
agg <- dcast.data.table(agg, Filename ~ variable + MBP)
agg[,Max_RAM_GB_mbp11_0_09 := as.numeric(Max_RAM_GB_mbp11_0_09)]
agg[,Max_RAM_GB_mbp11_0_10 := as.numeric(Max_RAM_GB_mbp11_0_10)]
agg[,RAM_diff := Max_RAM_GB_mbp11_0_10 - Max_RAM_GB_mbp11_0_09]
#agg[,RAM_diff := round(RAM_diff / Max_RAM_GB_mbp11_0_09 * 100, 1)]
agg <- agg[order(RAM_diff),]
ggplot(agg, aes(x=Max_RAM_GB_mbp11_0_10, y=Max_RAM_GB_mbp11_0_09, color=Filename)) +
  geom_point() +
  scale_color_manual(values=colors) +
  theme_tufte(base_family="Helvetica") +
  theme(legend.position="none") +
  geom_abline(slope=1)
agg[,Max_RAM_GB_mbp11_0_09 := round(Max_RAM_GB_mbp11_0_09, 1)]
agg[,Max_RAM_GB_mbp11_0_10 := round(Max_RAM_GB_mbp11_0_10, 1)]
#agg[,RAM_diff := NULL]
agg

# Plot common models
agg <- dat[,list(
  Max_RAM_GB=max(Max_RAM_GB, na.rm=T)
), by=list(Filename, MBP, Sample_Pct, main_task)]
agg <- melt.data.table(agg, measure.vars=c('Max_RAM_GB'))
agg <- dcast.data.table(agg, Filename + main_task + Sample_Pct ~ MBP)
ggplot(agg, aes(x=mbp11_0_10, y=mbp11_0_09, color=Sample_Pct)) +
  geom_point() +
  theme_tufte(base_family="Helvetica") +
  theme(legend.position="bottom") +
  geom_abline(slope=1)
mod <- lm(mbp11_0_10 ~ 0 + mbp11_0_09, agg)
summary(mod)
