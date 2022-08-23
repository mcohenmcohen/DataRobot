rm(list=ls(all=T))
gc(reset=T)
library(data.table)
library(ggplot2)


old_raw = fread('http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests=5a81c81b8d97700001219f2f&max_sample_size_only=false')
new_raw = fread('http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests=5a81f025478a9e00014fefb3&max_sample_size_only=false')

# 59f2353bc79de00001ca8c31 - 4.0 benchmark
# 5a81c81b8d97700001219f2f - 4.0.4 benchmark
# 5a7df300e2779d00015ffb38 - original dockerized Mbtests
# 5a81f025478a9e00014fefb3 - mbtest of branch with more LGBM cores
# 5a831465dfc2a90001e56f6e - branch cherry pick into 4.0 with more LGBM cores

old = old_raw[,list(
	max_runtime_hours_4.0 = max(Total_Time_P1, na.rm=T) / 3600,
	RAM_GB_4.0 = max(Max_RAM, na.rm=T) / 1e9
), by = 'Filename']

new = new_raw[,list(
	max_runtime_hours_4.2 = max(Total_Time_P1, na.rm=T) / 3600,
	RAM_GB_4.2 = max(Max_RAM, na.rm=T) / 1e9
), by = 'Filename']

comp = merge(old, new, by='Filename', all=T)

comp[,ratio := max_runtime_hours_4.2 / max_runtime_hours_4.0]
comp[,diff := max_runtime_hours_4.2 - max_runtime_hours_4.0]

setorderv(comp, 'ratio')
ggplot(comp, aes(x=max_runtime_hours_4.0, y=max_runtime_hours_4.2)) +
  geom_point() + geom_abline(slope=1, intercept=0) + theme_bw()

out = comp[,list(
	Filename,
	x4.0 = max_runtime_hours_4.0,
	x4.2 = max_runtime_hours_4.2,
	ratio
	# diff
)]
out = data.frame(out)
row.names(out) = out[['Filename']]
out[['Filename']] = NULL
round(out, 2)
round(out[out$ratio > 1.5 | 1/out$ratio > 1.5,], 2)
