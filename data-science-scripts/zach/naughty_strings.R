rm(list=ls(all=T))
gc(reset=T)
library(jsonlite)
library(httr)
library(stringi)
library(pbapply)
library(readr)
set.seed(42)

# Helpers
random_sentences = function(vocab, number, min_words=5, max_words=25){
  out = sapply(1:number, function(x) sample(min_words:max_words, 1))
  out = pbsapply(out, function(x) stri_paste(sample(vocab, x), collapse=' '))
  return(out)
}
random_data = function(number, vocab, t){
  data.frame(
    text = random_sentences(vocab, N),
    cat = sample(vocab, N, replace=T),
    target = t
  )
}

# Get bad data
# https://github.com/minimaxir/big-list-of-naughty-strings
url = 'https://raw.githubusercontent.com/minimaxir/big-list-of-naughty-strings/master/blns.json'
data = GET(url)
bad_words = fromJSON(content(data))
N = length(bad_words)

# Random good data
# From http://listofrandomwords.com/index.cfm?blist
good_words = c(
  'brownsburg',
  'travelable',
  'unteachable',
  'keith',
  'asynergy',
  'redisciplined',
  'supermilitary',
  'afternoon',
  'hydrated',
  'apostrophic',
  'interfertile',
  'pseudoaquatic',
  'electrotyped',
  'firecracker',
  'subeditor',
  'glaive',
  'hoodedness',
  'aboardage',
  'perikiromene',
  'gestate',
  'turbidimetric',
  'precyclone',
  'lilac',
  'unstriving',
  'uncontemptibleness',
  'hereditist',
  'vitamin',
  'uncasked',
  'doppelgï¿¥ï¾ nger',
  'shakespearean',
  'palisado',
  'entrapment',
  'airsick',
  'lao',
  'hopei',
  'pelopidae',
  'trudgen',
  'winterfed',
  'garment')

# Make dataset
N <- 10000
bad_data <- random_data(N, bad_words, 1)
good_data <- random_data(N, good_words, 0)
out <- rbind(bad_data, good_data)
out <- sample(out)
write_csv(out, '~/datasets/naughy_strings.csv')