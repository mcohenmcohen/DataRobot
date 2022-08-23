library(text2vec)
library(readr)
library(stringi)
library(stringr)
library(pbapply)
library(Matrix)
library(data.table)
set.seed(42)

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

#Define reader and tokenizer for articles
reader <- function(x){
  out <- read_lines(x)
  out <- out[nchar(out) > 4]
  return(out)
}
tokenizer <- function(x) strsplit(x, "", fixed=TRUE)
read_and_tokenize <- function(x){itoken(ifiles(x, reader_function = reader), tokenizer = tokenizer)}

#Create vocabulary - 4grams
#http://dsnotes.com/articles/text2vec-0-3
#http://dsnotes.com/articles/glove-enwiki
v <- create_vocabulary(read_and_tokenize(fls), ngram=c(ngram_min = 1L, ngram_max = 4L))
v_prune <- prune_vocabulary(v, term_count_min = 10L, doc_proportion_max=.50)
saveRDS(v_prune, file='~/datasets/english_char_vocab.RDS')
#v_prune <- readRDS('~/datasets/english_char_vocab.RDS')

#Create corpus and tcm:
ts <- read_and_tokenize(fls)
corpus <- create_corpus(ts, vectorizer = vocab_vectorizer(vocabulary=v_prune, grow_dtm = FALSE, skip_grams_window = 4L))
tcm <- get_tcm(corpus)
image(tcm[letters,letters])
saveRDS(tcm, file='~/datasets/english_char_tcm.RDS')

#Fit GloVe model
DIM = 600
X_MAX = 100L
WORKERS = 4L
NUM_ITERS = 20L
CONVERGENCE_THRESHOLD = 0.005
LEARNING_RATE = 0.15
RcppParallel::setThreadOptions(numThreads = WORKERS)

fit <- glove(
  tcm = tcm,
  word_vectors_size = DIM,
  num_iters = NUM_ITERS,
  learning_rate = LEARNING_RATE,
  x_max = X_MAX,
  shuffle_seed = 42L,
  # we will stop if global cost will be reduced less then 0.5% then previous SGD iteration
  convergence_threshold = CONVERGENCE_THRESHOLD)
saveRDS(fit, file='~/datasets/english_char_GloVe_model.RDS')

#Pull out most similar "words"
row_wise_norm <- function(m) {
  d <- Diagonal(x=1/sqrt(rowSums(m^2)))
  return(as.matrix(t(crossprod(m, d))))
}
words <- rownames(tcm)
m <- fit$word_vectors$w_i + fit$word_vectors$w_j
rownames(m) <-  words
m <- row_wise_norm(m)

#Look at sim for a workd
sim <- tcrossprod(m['e_g',], m)
diag(sim) <- 0
sim <- data.table(word=colnames(sim), sim=sim[1,])
setkeyv(sim, 'sim')
sim <- sim[sim > 0 & sim < 1,]
head(sim[order(sim, decreasing=TRUE),], 10)
