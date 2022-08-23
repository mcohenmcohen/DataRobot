stop()
rm(list=ls(all=T))
gc(reset=T)

library(data.table)
library(stringi)
library(text2vec)
library(Matrix)
library(pbapply)
library(irlba)
library(glmnet)
library(caTools)
library(Metrics)

# Load raw data
train_raw <- fread('https://s3.amazonaws.com/datarobot_public_datasets/crowd_text_train_80.csv')
test_raw <- fread('https://s3.amazonaws.com/datarobot_public_datasets/crowd_text_test_20.csv')

# Tokenize categoricals
prep_fun = stri_trans_tolower
tok_fun = word_tokenizer

tokenize_train_test <- function(a, b){
  
  a <- as.character(a)
  b <- as.character(b)

  it_train = itoken(
    a, 
    preprocessor = prep_fun, 
    tokenizer = tok_fun, 
    progressbar = FALSE)
  it_test = itoken(
    b, 
    preprocessor = prep_fun, 
    tokenizer = tok_fun, 
    progressbar = FALSE)
  
  vocab = create_vocabulary(it_train, ngram = c(1L, 2L))
  pruned_vocab = prune_vocabulary(
    vocab, 
    term_count_min = 10, 
    doc_proportion_max = 0.5,
    doc_proportion_min = 0)
  vectorizer = vocab_vectorizer(pruned_vocab)
  
  dtm_train = create_dtm(it_train, vectorizer)
  dtm_test = create_dtm(it_test, vectorizer)
  
  dtm_train = normalize(sign(dtm_train), norm='l2')
  dtm_test = normalize(sign(dtm_test), norm='l2')
  
  return(list(dtm_train, dtm_test))
}

target <- 'median_relevance'
cats <- names(which(sapply(train_raw, class) == 'character'))
#cats <- c(cats, 'item_condition_id')
nums <- setdiff(names(train_raw), c(cats, target))

cat_mats <- pblapply(cats, function(x){
  tokenize_train_test(train_raw[[x]], test_raw[[x]])
})

cat_mats_train = lapply(cat_mats, '[[', 1)
cat_mats_test = lapply(cat_mats, '[[', 2)

cat_mats_train = Reduce(cbind, cat_mats_train)
cat_mats_test = Reduce(cbind, cat_mats_test)

cat_mats_train = normalize(sign(cat_mats_train), norm='l2')
cat_mats_test = normalize(sign(cat_mats_test), norm='l2')

stopifnot(ncol(cat_mats_train) == ncol(cat_mats_test))
stopifnot(nrow(cat_mats_train) == nrow(train_raw))
stopifnot(nrow(cat_mats_test) == nrow(test_raw))

# "center" and scale numerics
nums_train = as.matrix(train_raw[,nums,with=F])
nums_test = as.matrix(test_raw[,nums,with=F])

for(i in 1:ncol(nums_train)){
  center <- median(nums_train[,i])
  scale <- sd(nums_train[,i])
  
  nums_train[,i][is.na(nums_train[,i])] <- center
  nums_test[,i][is.na(nums_test[,i])] <- center
  
  nums_train[,i] <- (nums_train[,i] - center) / scale
  nums_test[,i]  <- (nums_test[,i] - center) / scale

  # bins <- sort(unique(quantile(nums_train[,i], 0:100/100)))
  # 
  # f <- approxfun(bins, seq_along(bins)/length(bins), rule = 2)
  # 
  # nums_train[,i] <- f(nums_train[,i])
  # nums_test[,i]   <- f(nums_train[,i])
  
}

# Combine all together
train = cbind(Matrix(nums_train, sparse=T), cat_mats_train)
test  = cbind(Matrix(nums_test, sparse=T), cat_mats_test)

stopifnot(ncol(train) == ncol(test))
stopifnot(nrow(train) == nrow(train_raw))
stopifnot(nrow(test) == nrow(test_raw))

# SVD via irlba
model_svd = irlba(train, nv=64, nu=0, verbose=T)
train_svd = train %*% model_svd$v
test_svd  = test %*% model_svd$v

# Interactions
train_poly = matrix(0, nrow=nrow(train), ncol=ncol(train_svd)^2)
test_poly  = matrix(0, nrow=nrow(test) , ncol=ncol(train_svd)^2)
col=1
for(i in 1:ncol(train_svd)){
  for(j in i:ncol(train_svd)){
    train_poly[,col] <- train_svd[,i] * train_svd[,j]
    test_poly[,col]  <- test_svd[,i] * test_svd[,j]
    col = col + 1
  }
}
# train_poly <- Matrix(cbind(train_svd, train_poly), sparse=T)
# test_poly <- Matrix(cbind(test_svd, test_poly), sparse=T)
train_poly <- Matrix(cbind(train_svd, train_poly), sparse=T)
test_poly <- Matrix(cbind(test_svd, test_poly), sparse=T)

# Final data
train_final <- cbind(train, train_svd)
test_final  <- cbind(test, test_svd)
train_target <- train_raw[[target]]
test_target <- test_raw[[target]]

stopifnot(ncol(train_final) == ncol(test_final))
stopifnot(nrow(train_final) == nrow(train_raw))
stopifnot(nrow(test_final) == nrow(test_raw))

# Model
library(doParallel)
NFOLD <- 10
registerDoParallel(NFOLD)
model <- cv.glmnet(train_final, train_target, family="gaussian", alpha=1, nfolds=NFOLD, parallel=T)

# Predict
s <- "lambda.min"
p_train <- predict(model, train_final, s = s, type='response')
p_test  <- predict(model, test_final, s = s, type='response')

# Best Kaggle: 0.72189  DR: 0.759993335
rmse(train_target, p_train)
rmse(test_target, p_test)  # 0.8360819
