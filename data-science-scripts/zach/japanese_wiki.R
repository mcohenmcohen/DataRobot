library(text2vec)
library(readr)
library(stringi)
library(stringr)
library(Matrix)

set.seed(42)

#Define reader and tokenizer
reader <- function(x){
  out <- read_lines(x)
  out <- stri_trim(stri_replace_all_regex(out, '[\\{C}|\\{Z}|\\{M}]', ' '))
  out <- out[out!='']
  return(out)
}
tokenizer <- function(x) strsplit(x, "", fixed=TRUE)
read_and_tokenize <- function(x){itoken(ifiles(x, reader_function = reader), tokenizer = tokenizer)}

start <- Sys.time()
#system("./WikiExtractor.py ~/datasets/jawiki-latest-pages-articles.xml.bz2 -o ~/datasets/jawiki --no-templates -ns Main --no-doc --no-title --processes 9")

#Make list of articles
dirs <- list.files("~/datasets/jawiki", full.names = T)
fls <- unname(unlist(sapply(dirs, function(x) paste0(x, '/', list.files(x)))))
#fls <- sample(fls, 1000)

#Create vocabulary
#http://dsnotes.com/articles/text2vec-0-3
#http://dsnotes.com/articles/glove-enwiki
v <- create_vocabulary(read_and_tokenize(fls), ngram=c(ngram_min = 1L, ngram_max = 5L))
v <- prune_vocabulary(v, term_count_min = 10, doc_proportion_max = 0.2)
saveRDS(v, file='~/datasets/japanese_vocab.RDS')

#Create corpus and tcm:
ts <- read_and_tokenize(fls)
corpus <- create_corpus(ts, vectorizer = vocab_vectorizer(vocabulary=v, grow_dtm = FALSE, skip_grams_window = 10L))
tcm <- get_tcm(corpus)
saveRDS(tcm, file='~/datasets/japanese_tcm.RDS')

#Prune TCM
ind <- tcm@x >= 1
tcm@x <- tcm@x[ind]
tcm@i <- tcm@i[ind]
tcm@j <- tcm@j[ind]

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
saveRDS(fit, file='~/datasets/japanese_GloVe_model.RDS')

#Timing:
finish <- Sys.time()
print(finish - start)

#Lookup most similar words for top 100 most frequenty words
vocab_sort <- v$vocab[order(doc_counts, decreasing=TRUE),][,terms]
most_freq <- head(vocab_sort, 10)

#Pull out word vectors and sort by vocab
row_wise_norm <- function(m) {
  d <- Diagonal(x=1/sqrt(rowSums(m^2)))
  return(as.matrix(t(crossprod(m, d))))
}
words <- rownames(tcm)
m <- fit$word_vectors$w_i + fit$word_vectors$w_j
rownames(m) <-  words
m <- m[vocab_sort,]
m <- row_wise_norm(m)
m <- row_wise_norm(m)

#Look at sim for most frequent words
sim <- tcrossprod(m[1:100,], m)
diag(sim) <- 0
best <- apply(sim, 1, function(x)names(x)[which.max(x)])
best


sim <- tcrossprod(m["水",], m)


