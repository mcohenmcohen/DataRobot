
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
# SLC14_2_dirty - 10_0_08
########################################################

# SVM Forest @100%
blueprint_info("5769dd3e34c997a0d370911b")[,1:5,with=FALSE]
blueprint_json("5769dd3e34c997a0d370911b")

########################################################
# AirBnB - 10_0_08
########################################################

# Blender at @100%
blueprint_info("576a84fb34c99711d6333777")[,1:5,with=FALSE]

########################################################
# GSOD - 10_0_08
########################################################

# Linear at @100%
blueprint_info("576920d034c997a0d3708fdf")[,1:5,with=FALSE]

# Blender at @100%
blueprint_info("57688a2634c99720171a1214")[,1:5,with=FALSE]

########################################################
# AirBnB - 10_0_06 float32
########################################################

# Ridge at @100%
blueprint_info("575433c834c9970b8d707094")[,1:5,with=FALSE]

# Blender @95%
out <- blueprint_info("575433e834c9970b8d707098")[,1:5,with=FALSE]
out[,list(
  fit.max.RAM.GB = round(max(fit.max.RAM.GB), 1),
  transform.max.RAM.GB = round(max(transform.max.RAM.GB), 1),
  fit.clock.time.hours = round(sum(fit.clock.time.hours),1),
  transform.clock.time.hours = round(sum(transform.clock.time.hours), 1)
), by='task_name']
out[,id := 1:.N, by='task_name']
ggplot(out, aes(x=id, y=fit.max.RAM.GB, col=task_name)) + geom_line() + theme_bw()
ggplot(out[task_name=='SCTXT2',], aes(x=id, y=fit.max.RAM.GB, col=task_name)) + geom_line() + theme_bw()
ggplot(out[task_name=='SCTXT2',], aes(x=id, y=transform.max.RAM.GB, col=task_name)) + geom_line() + theme_bw()

#Prime @95%
out <- blueprint_info("57548f7534c9970b8d7070bc")[,1:5,with=FALSE]
out[,list(
  fit.max.RAM.GB = round(max(fit.max.RAM.GB), 1),
  transform.max.RAM.GB = round(max(transform.max.RAM.GB), 1),
  fit.clock.time.hours = round(sum(fit.clock.time.hours),1),
  transform.clock.time.hours = round(sum(transform.clock.time.hours), 1)
), by='task_name']

########################################################
# GSOD - 10_0_06 float32
########################################################

# Prime @ 95%
blueprint_info("57548f7734c9970b8d7070c0")[,1:5,with=FALSE]

# Enet @ 100%
blueprint_info("5752e35334c9970b8d706f8f")[,1:5,with=FALSE]

########################################################
# SLC14 - 10_0_06 float32
########################################################

# Prime @ 95%
blueprint_info("57548f7434c9970b8d7070b8")[,1:5,with=FALSE]

# Blender @95%
out <- blueprint_info("5754339634c9970b8d707090")[,1:5,with=FALSE]
out[,list(
  fit.max.RAM.GB = round(max(fit.max.RAM.GB), 1),
  transform.max.RAM.GB = round(max(transform.max.RAM.GB), 1),
  fit.clock.time.hours = round(sum(fit.clock.time.hours),1),
  transform.clock.time.hours = round(sum(transform.clock.time.hours), 1)
), by='task_name']
out[,id := 1:.N, by='task_name']
ggplot(out, aes(x=id, y=fit.max.RAM.GB, col=task_name)) + geom_line() + theme_bw()
ggplot(out[task_name=='SCTXT2',], aes(x=id, y=fit.max.RAM.GB, col=task_name)) + geom_line() + theme_bw()
ggplot(out[task_name=='SCTXT2',], aes(x=id, y=transform.max.RAM.GB, col=task_name)) + geom_line() + theme_bw()
