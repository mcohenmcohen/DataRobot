library(datarobot)
library(data.table)
project = GetProject('5bc0b538b64ee91f026dbef7')
ViewWebProject(project)
model = GetModel(project, '5bc0b566566ad60176107ff1')
pred_job = RequestPredictions(model, '~/datasets/iris_int_in_species.csv')
preds = GetPredictJob(project, pred_job)

library(httr)
library(jsonlite)
pd = fread('~/datasets/iris_int_in_species.csv')
pjson = toJSON(pd[150,])
pred = POST(
  'cfds-ccm-prod.orm.datarobot.com/predApi/v1.0/5bc0b538b64ee91f026dbef7/5bc0b566566ad60176107ff1/predict', 
  body=pjson, 
  content_type_json(),
  add_headers('datarobot-key' = ""))
status_code(pred)
