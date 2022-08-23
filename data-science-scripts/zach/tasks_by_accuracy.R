library(data.table)
library(stringi)
library(pbapply)
library(jsonlite)

fixPythonJSON <- function(t, ...) {
  t <- stringi::stri_replace_all_regex(t, "u'(.*?)'", '"$1"')
  t <- stringi::stri_replace_all_fixed(t, '""', '"')
  t <- stringi::stri_replace_all_fixed(t, '{"}', '')
  t <- stringi::stri_replace_all_fixed(t, ': None', ': null')
  t[is.na(t)] <- '{}'
  t[t==''] <- '{}'
  t
}

mbtestid = '577fc68182785c15b465dba7'

dat_raw = fread(
  paste0(
    'http://shrink.prod.hq.datarobot.com/api/leaderboard_export/advanced_export.csv?mbtests=',
    mbtestid,
    '&max_sample_size_only=true'
    )
  )

#Parse blueprint json
dat <- copy(dat_raw)
setnames(dat, make.names(names(dat)))
dat <- dat[is_blender == FALSE,]
dat[,Blueprint := fixPythonJSON(Blueprint)]
dat[,Blueprint := pblapply(Blueprint, fromJSON)]

#Make a list of all the blueprints for a given main task, just for fun
main_task_to_blueprint <- dat[,list(Blueprint = list(c(Blueprint))), by='main_task']
main_task_to_blueprint[,Blueprint := pblapply(Blueprint, toJSON, pretty=TRUE, flatten=TRUE)]

#Main task + sub tasks
dat[,simple_blueprint := paste(main_task, X_tasks)]

#Best
dat <- dat[,list(Filename, Gini.Norm_P1, Max_RAM, main_task, simple_blueprint)]
dat <- dat[,best := as.integer(Gini.Norm_P1 == max(Gini.Norm_P1)), by='Filename']

#Aggregate
agg <- dat[best == 1,]
agg <- agg[,list(
  .N,
  min_gini = min(Gini.Norm_P1, na.rm=T),
  med_gini = median(Gini.Norm_P1, na.rm=T),
  max_gini = max(Gini.Norm_P1, na.rm=T)
), by='simple_blueprint']
setorder(agg, N)

