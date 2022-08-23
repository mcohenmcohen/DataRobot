library(data.table)
library(readr)
library(stringi)

#aloi 1,000 classes -> 100 classes
set.seed(100)
aloi <- fread('https://s3.amazonaws.com/datarobot_public_datasets/libsvm/datasets/aloi.csv')
aloi[,target := paste0('c', target)]
classes_10 <- aloi[,sample(unique(target), 10)]
classes_100 <- aloi[,sample(unique(target), 100)]
aloi_10 <- aloi[target %in% classes_10,]
aloi_100 <- aloi[target %in% classes_100,]
write_csv(aloi_10, '~/aloi_10.csv')
write_csv(aloi_100, '~/aloi_100.csv')
write_csv(aloi, '~/aloi_1000.csv')
utils:::format.object_size(file.size('~/aloi_10.csv'), units='MB')
utils:::format.object_size(file.size('~/aloi_100.csv'), units='MB')
utils:::format.object_size(file.size('~/aloi_1000.csv'), units='MB')

# Reddit: 5GB -> 500MB; 25,000 classes -> 100 classes
reddit_raw <- read_csv('https://s3.amazonaws.com/datarobot_public_datasets/reddit_may_2015_5Gb.csv', quote='')
problems(reddit_raw)
reddit = as.data.table(reddit_raw)
reddit[,subreddit := stri_trans_tolower(subreddit)]
setkey(reddit, 'subreddit')

subreddits <- reddit[,list(.N), by='subreddit']
subreddits <- subreddits[order(N, decreasing = T),]
subreddits <- subreddits[!is.na(subreddit),]
subreddits <- subreddits[subreddit != '0',]
subreddits[,N := NULL]
subreddits_top_10 <- head(subreddits, 10)
subreddits_top_100 <- head(subreddits, 100)
subreddits_top_1000 <- head(subreddits, 1000)

setkeyv(subreddits_top_10, 'subreddit')
setkeyv(subreddits_top_100, 'subreddit')
reddit_top_10 <- reddit[subreddits_top_10,]
reddit_top_100 <- reddit[subreddits_top_100,]
reddit_top_1000 <- reddit[subreddits_top_1000,]

target = 500 / 5000 * nrow(reddit) * 500/537.4
set.seed(42)
reddit_top_10 <- reddit_top_10[sample(1:.N, target),]
reddit_top_100 <- reddit_top_100[sample(1:.N, target),]
reddit_top_1000 <- reddit_top_1000[sample(1:.N, target),]

write_csv(reddit_top_10, '~/reddit_top_10.csv')
write_csv(reddit_top_100, '~/reddit_top_100.csv')
write_csv(reddit_top_1000, '~/reddit_top_1000.csv')

utils:::format.object_size(file.size('~/reddit_top_10.csv'), units='MB')
utils:::format.object_size(file.size('~/reddit_top_100.csv'), units='MB')
utils:::format.object_size(file.size('~/reddit_top_1000.csv'), units='MB')
