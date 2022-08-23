
#Libraries
library(readr)
library(data.table)
library(stringi)

#Setup
x <- read_delim('~/datasets/yelp_review_polarity_full.csv', ',')
x <- data.table(x)
x[class_id == 1, class_id := -1]
x[class_id == 2, class_id := 1]
x[,text := stri_replace_all_fixed(text, ':', ' ')]
x[,text := stri_replace_all_fixed(text, '\n', '\t')]
x[,text := stri_replace_all_fixed(text, '\\n', '\t')]
x[,text := stri_replace_all_fixed(text, '|', ' ')]

#Save file
vw_file <- '~/datasets/vw_sentiment_train.txt'
cache_file <- paste0(vw_file, '.cache')
unlink(vw_file)
unlink(cache_file)
out <- x[,stri_paste(class_id, ' |t ', text)]
cat(out, file=vw_file, sep='\n')

#Run VW
system(paste('head', vw_file))
system(paste(
  'vw',
  '-d', vw_file,
  '--cache_file', cache_file,
  '-f vw_sentiment.vw',
  '--loss_function logistic',
  '--ngram 2 --nn 10 --passes 2 -b 22 --meanfield'
))

system(paste(
  'vw',
  '-d', vw_file,
  '-i vw_sentiment.vw',
  '-p ~/datasets/vw_preds',
  '-r ~/datasets/vw_preds.raw',
  '--loss_function logistic'
))

system('head  ~/datasets/vw_preds.raw')
