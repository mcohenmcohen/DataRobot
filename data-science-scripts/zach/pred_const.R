library(data.table)
main_dir <- '~/workspace/data-science-scripts/zach/pred_const_2_master/'
test_dir <- '~/workspace/data-science-scripts/zach/pred_const_2_branch/'
dat <- lapply(list.files(test_dir), function(x){
  master <- fread(paste0(main_dir, x))
  test <- fread(paste0(test_dir, x))
  setkeyv(master, 'row_id')
  setkeyv(test, 'row_id')
  compare <- names(master)[grepl('Prediction', names(master), fixed=T)]
  diff <- abs(master[,compare,with=F] - test[,compare,with=F])
  diff <- apply(diff, 1, max)
  out <- data.table(
    file=x,
    diff=diff
  )
})
dat <- rbindlist(dat)
summary(dat)
