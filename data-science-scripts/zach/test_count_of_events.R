library(datarobot)
library(data.table)
pr = datarobot::GetProject('59f09cdac8089174a44181aa')
pred_data = datarobot::UploadPredictionDataset(pr, 'https://s3.amazonaws.com/datarobot_public_datasets/InsuranceDemoWithClaimCount.csv')
preds = datarobot::RequestPredictionsForDataset(pr, '59f09e1fb2ca1a276a754094', pred_data$id)
pred_res = datarobot::GetPredictions(pr, preds)
head(pred_res)


pred_dat <- data.table::fread('https://s3.amazonaws.com/datarobot_public_datasets/InsuranceDemoWithClaimCount.csv')
pred_dat_small <- head(pred_dat, 10)
pred_dat_small[,IncurredClaims:=NULL]
pred_dat_small[,ClaimCount:=NULL]
pred_data2 = datarobot::UploadPredictionDataset(pr, pred_dat_small)
preds2 = datarobot::RequestPredictionsForDataset(pr, '59f09e1fb2ca1a276a754094', pred_data2$id)
pred_res2 = datarobot::GetPredictions(pr, preds2)
pred_res2
all(pred_res2 == pred_res[seq_along(pred_res2)])
