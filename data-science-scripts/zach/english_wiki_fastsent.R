library(text2vec)
library(readr)
library(stringi)
library(stringr)
library(pbapply)
library(Matrix)
library(data.table)
library(readr)
library(RcppCNPy)
set.seed(42)
setwd('~/datasets/')
setwd('~/workspace/data-science-scripts/zach/')

#Extract articles from html
#http://medialab.di.unipi.it/wiki/Wikipedia_Extractor
#https://github.com/attardi/wikiextractor
#https://dumps.wikimedia.org/enwiki/latest/
#Retrieved 2016-07-06
#7.829555556 hours to run the extractor with 9 processes
#system("./WikiExtractor.py ~/datasets/enwiki-latest-pages-articles.xml.bz2 -o ~/datasets/enwiki2 --no-templates -ns Main --no-doc --no-title --processes 9")

#Make list of articles
dirs <- list.files("~/datasets/enwiki", full.names = T)
fls <- unname(unlist(sapply(dirs, function(x) paste0(x, '/', list.files(x)))))

#Chunk list of files
set.seed(42)
fls <- sample(fls)
fls_chunk <- split(fls, (1:length(fls)) %% (floor(length(fls) / 10)))
table(sapply(fls_chunk, length))
cmds <- sapply(fls_chunk, function(x){
  cmd <- paste(x, collapse=' ')
  cmd <- paste('cat', cmd, '>> /Users/zachary/datasets/en_wikifull.txt')
  return(cmd)
})

#Concat articles
system("echo '' > /Users/zachary/datasets/en_wikifull.txt")
system("head /Users/zachary/datasets/en_wikifull.txt")
sink <- pblapply(cmds, system)
system("head /Users/zachary/datasets/en_wikifull.txt")

#Insert newlines after certain punctuation marks
if(FALSE){ #Run on system
  cp en_wikifull.txt en_wikifull_pp.txt #Copy in the system
  sed -i.bu -- 's/etc\./etc/g' en_wikifull_pp.txt
  sed -i.bu -- $'s/\. /\.\\\n/g' en_wikifull_pp.txt
  sed -i.bu -- $'s/\! /\!\\\n/g' en_wikifull_pp.txt
  sed -i.bu -- $'s/\? /\?\\\n/g' en_wikifull_pp.txt
}

#Fit Fastsent model
setwd('~/workspace/data-science-scripts/zach')
system('./fit_fastsent.py -i /Users/zachary/datasets/en_wikifull_pp.txt -o /Users/zachary/datasets/enwiki_fastsent')

#Data for inference
dat <- fread('https://s3.amazonaws.com/datarobot_data_science/text_data/rottentomatoes_sentiment.csv')
write.csv(dat[,Phrase], row.names=FALSE, '~/datasets/rottentomatoes_sentiment_notarget.csv')

#Infer Fastsent model
system('./infer_fastsent.py -i /Users/zachary/datasets/rottentomatoes_sentiment_notarget.csv -o /Users/zachary/datasets/tomatoes_to_wiki.npy -m /Users/zachary/datasets/enwiki_fastsent')

#Load results
mat <- npyLoad('/Users/zachary/datasets/tomatoes_to_wiki.npy')
mat <- data.table(mat)
out <- data.table(
  class_id = dat$Sentiment,
  mat
)

#Save to DR
write.csv(out, '~/datasets/tomatoes_fastsent_norm_full.csv', row.names=FALSE)
