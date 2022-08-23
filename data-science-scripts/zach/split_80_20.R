library(data.table)

for(f in list(
  'https://s3.amazonaws.com/datarobot_public_datasets/Depot_text_cosine_sim.csv',
  'https://s3.amazonaws.com/datarobot_public_datasets/amazon_text_cosine_sim.csv'
  #'https://s3.amazonaws.com/datarobot_public_datasets/bloggers_text_cosine_sim.csv'
)){
  dat <- fread(f, quote='')
  
  set.seed(42)
  idx <- runif(nrow(dat)) <= .80
  dat_train <- dat[which( idx),]
  dat_test  <- dat[which(!idx),]
  
  fout <- gsub(
    'https://s3.amazonaws.com/datarobot_public_datasets/',
    '~/datasets/',
    f,
    fixed=T
  )
  fout <- gsub('.csv', '', fout, fixed=T)
  fwrite(dat_train, paste0(fout, '_80.csv'))
  fwrite(dat_test, paste0(fout, '_20.csv'))
}