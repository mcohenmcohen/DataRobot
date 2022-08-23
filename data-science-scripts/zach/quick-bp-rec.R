library(mongolite)
library(jsonlite)
library(jsonlite)

USER = "zach"
PASS = "cfb35178d3654cbe8a9a6fdbc62892b8"
HOST = "prod-mongo-hidden.infra.ent.datarobot.com"

con <- mongolite::mongo(
  db = 'MMApp',
  collection = 'leaderboard',
  url=sprintf("mongodb://%s:%s@%s", USER, PASS, HOST), verbose=F)


lid <- '60cce5572740cdec3c729d96'  # complex
lid <- '60cce5612740cdec3c729dba'  # simple

con$find(query=paste0('{"_id": {"$oid":"', lid, '"}}'), limit=1)

filter <- '"pid":1, "samplesize":1, "blueprint":1, "cv_scores": 1, "time.start_time.reps=1" : 1'
it <- con$iterate(
  query=paste0(
    '{',
#    '"_id": {"$oid":"', lid, '"},',
    '"time.start_time.reps=1": {"$gt": 1609477200},', #unix timestamp for jan 1 of 2021
    '"time.start_time.reps=1": {"$lt": 1617249600}', #unix timestamp for april 1 of 2021
    '}'),
  fields = paste0('{', filter, ', "_id": false}'), limit=1)
res <- it$json()
gsub('"', "'", res)
