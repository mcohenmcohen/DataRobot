library(rmongodb)
library(jsonlite)

blender_lids <- function(lid, host="mongo-0.prod.aws.datarobot.com"){
  library(rmongodb)

  mongo <- mongo.create(host, 'rs0')
  mongo.is.connected(mongo)
  query <- mongo.bson.from.list(list(
    '_id' = mongo.oid.from.string(lid)
  ))
  filter <- '{"blender" : "1"}'
  res <- mongo.find.one(mongo, "MMApp.leaderboard", query, filter)
  mongo.destroy(mongo)
  res <- mongo.bson.to.list(res)$blender$inputs
  res <- sapply(res, "[[", 'lid')
  return(res)
}

blueprint_json <- function(lid, host="mongo-0.prod.aws.datarobot.com"){
  library(rmongodb)

  mongo <- mongo.create(host, 'rs0')
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

avg_blender_lid <- '57da9d30300cc3695a7521b5'
blended_models_lids <- blender_lids(avg_blender)
blueprints <- lapply(blended_models_lids, blueprint_json)
print(blueprints)
