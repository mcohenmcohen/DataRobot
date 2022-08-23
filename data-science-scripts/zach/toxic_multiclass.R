stop()
rm(list=ls(all=T))
gc(reset=T)
library(data.table)
library(jsonlite)
library(stringi)
library(pbapply)

train <- fread('/Users/zachary/Downloads/jigsaw-toxic-comment-classification-challenge/train.csv')
test <- fread('/Users/zachary/Downloads/jigsaw-toxic-comment-classification-challenge/test.csv')

colnames <- c('toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate')
target_list <- lapply(colnames, function(t){
  ifelse(train[[t]]==1, stri_paste('"', t, '"'), "REMOVE")
})
target <- stri_paste(
  target_list[[1]],
  target_list[[2]],
  target_list[[3]],
  target_list[[4]],
  target_list[[5]],
  target_list[[6]],
  sep=', ', ignore_null=T
)

target <- stri_replace_all_fixed(target, "REMOVE, ", "")
target <- stri_replace_all_fixed(target, ", REMOVE", "")
target <- stri_replace_all_fixed(target, "REMOVE", "")
target <- stri_paste("[", target, "]")

validate <- pbsapply(target, fromJSON)
target[7]

train[,target := target]
fwrite(train[,list(comment_text, target)], '~/Downloads/jigsaw-toxic-comment-classification-challenge.csv')

samp_sub <- fread('/Users/zachary/Downloads/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')
pred <- fread('/Users/zachary/Downloads/jigsaw-toxic-comment-classification-challenge.csv_Keras_Deep_Self-Normalizing_Residual_Neural_Networ_(8)_64.0_619570690b9b595a88fec07a_04ccd842-2600-4da5-b062-2_test.csv')
setorderv(pred, 'row_id')
setnames(pred, names(samp_sub))
pred[,id := samp_sub[['id']]]
fwrite(pred, '~/Downloads/jigsaw-pred.csv')
