
########################################################
# Libraries
########################################################

library(data.table)
library(rmongodb)
library(jsonlite)
library(ggplot2)

########################################################
# Useful functions
########################################################

task_info_table <- function(res){
  library(data.table)
  library(rmongodb)
  x <- mongo.bson.to.list(res)[['task_info']][[1]]
  x <- lapply(x, function(a){
    out <- a[[1]]
    out <- out[!(sapply(out, is.null))]
    out <- data.table(data.frame(out))
    setnames(out, gsub("predict", "transform", names(out)))
    return(out)
  })

  out <- rbindlist(x, fill=TRUE, use.names=TRUE)
  #print(out)
  for(n in c(
    'fit.max.RAM', 'fit.avg.RAM', 'transform.max.RAM', 'fit.total.RAM',
    'transform.avg.RAM', 'transform.total.RAM', 'fit.CPU.time',
    'fit.clock.time', 'transform.clock.time', 'transform.CPU.time',
    'transform.sys.time', 'fit.sys.time')){
    if(! n %in% names(out))
      set(out, j=n, value=as.numeric(NA))
  }

  out[, fit.max.RAM := fit.max.RAM / 1e9]
  out[, fit.avg.RAM := fit.avg.RAM / 1e9]
  out[, transform.max.RAM := transform.max.RAM / 1e9]
  out[, fit.total.RAM := fit.total.RAM / 1e9]
  out[, transform.avg.RAM := transform.avg.RAM / 1e9]
  out[, transform.total.RAM := transform.total.RAM / 1e9]

  out[, fit.CPU.time := fit.CPU.time / 3600]
  out[, fit.clock.time := fit.clock.time / 3600]
  out[, transform.clock.time := transform.clock.time / 3600]
  out[, transform.CPU.time := transform.CPU.time / 3600]
  out[, transform.sys.time := transform.sys.time / 3600]
  out[, fit.sys.time := fit.sys.time / 3600]

  setnames(out, gsub(".RAM", ".RAM.GB", names(out), fixed=TRUE))
  setnames(out, gsub(".time", ".time.hours", names(out), fixed=TRUE))

  first <- c('task_name', 'fit.max.RAM.GB', 'transform.max.RAM.GB', 'fit.clock.time.hours', 'transform.clock.time.hours')
  setcolorder(out, c(first, setdiff(names(out), first)))

  return(out)
}

blueprint_info <- function(lid, host="10.20.53.43"){
  library(rmongodb)

  mongo <- mongo.create(host)
  mongo.is.connected(mongo)
  query <- mongo.bson.from.list(list(
    '_id' = mongo.oid.from.string(lid)
  ))
  filter <- '{"task_info":"1"}'
  res <- mongo.find.one(mongo, "MMApp.leaderboard", query, filter)
  mongo.destroy(mongo)
  return(task_info_table(res))
}

blueprint_json <- function(lid, host="10.20.53.43"){
  library(rmongodb)

  mongo <- mongo.create(host)
  mongo.is.connected(mongo)
  query <- mongo.bson.from.list(list(
    '_id' = mongo.oid.from.string(lid)
  ))
  filter <- '{"blueprint":"1"}'
  res <- mongo.find.one(mongo, "MMApp.leaderboard", query, filter)
  mongo.destroy(mongo)
  res <- mongo.bson.to.list(res)
  res[[1]] <- NULL
  res <- toJSON(res, pretty=TRUE)
  return(res)
}

########################################################
# Grockit OOM
########################################################
a <- blueprint_info("58c020d434c9971faa0d075d", '10.20.53.43')[,1:4,with=FALSE]
blueprint_json("58c020d434c9971faa0d075d", '10.20.53.43')
a

########################################################
# Owen's world MSD that ran longer than expected
########################################################
a <- blueprint_info("58becbc134c99768bdd797f8", '10.20.53.43')[,1:4,with=FALSE]
blueprint_json("58becbc134c99768bdd797f8", '10.20.53.43')
a

########################################################
# XGBoost + diff3 features owen's world
########################################################
a <- blueprint_info("58becb9634c99768bdd797df", '10.20.53.43')[,1:4,with=FALSE]
blueprint_json("58becb9634c99768bdd797df", '10.20.53.43')
a

########################################################
# XGBoost + unsupervised features owen's world
########################################################
a <- blueprint_info("58becb9634c99768bdd797de", '10.20.53.43')[,1:4,with=FALSE]
blueprint_json("58becb9634c99768bdd797de", '10.20.53.43')
a

########################################################
# "3.0" on owen's world
########################################################
a <- blueprint_info("58bb3de434c9970cb9ca9522", '10.20.53.43')[,1:4,with=FALSE]
blueprint_json("58bb3de434c9970cb9ca9522", '10.20.53.43')

########################################################
# "3.0"
########################################################
a <- blueprint_info("58b904fe10b418015ec3e5aa", 'shrink-mongo.dev.hq')[,1:4,with=FALSE]
blueprint_json("58b904fe10b418015ec3e5aa", 'shrink-mongo.dev.hq')
#
# `db <- mongo.create(host='10.50.228.186')` (edited)
#
# [9:41]
# `mongo_client = pymongo.MongoClient('shrink-mongo.dev.hq.datarobot.com',read_preference=ReadPreference.SECONDARY_ONLY)`

########################################################
# Owen's World cred1b1
########################################################

#64%
blueprint_info("57e9413634c9979c5789d79c", '10.20.53.43')[,1:4,with=FALSE]

#100%
blueprint_info("57f2898b34c9971fe7ab9478", '10.20.53.43')[,1:4,with=FALSE]

########################################################
# Owen's World XGBoost DIFF3 issues
########################################################

#Non Cython
blueprint_info("57d97c5a34c9972a8cd7238c", '10.20.53.43')[,1:4,with=FALSE]

#Cython
blueprint_info("57e3208a34c9979ba9a26b90", '10.20.53.43')[,1:4,with=FALSE]

#blueprint_json("57e3208a34c9979ba9a26b90", '10.20.53.43')

########################################################
# "2p9" running 2.9.0 - 10_0_10 I think
########################################################
b2.9 <- blueprint_info("57d21513c9ac20006ed70d69", '10.50.182.186')[,1:4,with=FALSE]
#blueprint_json("57d21513c9ac20006ed70d69", '10.50.182.186')


########################################################
# RD2 running 2.8.2 - 10_0_08
########################################################

#http://10.20.55.117/projects/579cfcdb1f4f77003f3554c1/models/57a0dc361f4f7700b1a7f678
#25% 57a09a321f4f7700828fa768
#50% 57a09a3c1f4f7700b242e0fd
#85% 57a09a511f4f770066411c10
#87% 57a0dc361f4f7700b1a7f678

# XGBoost with DIFF
b2.8 <- blueprint_info("57a0dc361f4f7700b1a7f678", '10.20.55.117')[,1:4,with=FALSE]
b2.8
#blueprint_json("57a0dc361f4f7700b1a7f678", '10.20.55.117)

########################################################
# Owen's World running 2.9 - 10_0_08
########################################################
#http://10.20.53.43/projects/57a099c434c9974a1c29ba3a/models/57a0daa034c9974a1c29bb20

# 75% 57a0b4a834c9974a1c29bb11
# 85% 57a0d90434c9974a1c29bb18
# 87% 57a099c434c9974a1c29ba3a
# 95% 57a0b48534c9974a1c29bb05

# XGBoost with DIFF
b2.9 <- blueprint_info("57a0daa034c9974a1c29bb20", '10.20.53.43')[,1:4,with=FALSE]
b2.9
#blueprint_json("57a0daa034c9974a1c29bb20", '10.20.53.43')
#blueprint_json("57a0dc361f4f7700b1a7f678", '10.20.55.117)

########################################################
# Compare
########################################################

out <- b2.8
for(v in names(out)[2:4]){
  set(out, j=v, value = round(b2.9[[v]] / b2.8[[v]], 2))
}
out
