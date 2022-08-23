library(data.table)
dat_raw = fread('http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests=5e06230fdad22b000d0a5264&max_sample_size_only=false')

dat = dat_raw[Filename=='10k_diabetes_80.xlsx',list(main_task, `Prediction LogLoss`)]
dat = dat[!is.na(`Prediction LogLoss`),][order(`Prediction LogLoss`),]
dat
