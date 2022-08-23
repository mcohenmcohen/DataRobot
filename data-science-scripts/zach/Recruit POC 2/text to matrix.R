library(data.table)
library(tau)
library(stringi)
library(pbapply)
library(Matrix)
dat <- fread('data/lt0_v5tv.csv')
texts <- dat$Job.Description.Detailed

find_ngrams <- function(dat, n, verbose=FALSE){
  stopifnot(is.list(dat))
  stopifnot(is.numeric(n))
  stopifnot(n>0)
  if(n == 1) return(dat)

  APPLYFUN <- lapply
  if(verbose){
    APPLYFUN <- pblapply
  }

  APPLYFUN(dat, function(y) {
    if(length(y)<=1) return(y)
    c(y, unlist(lapply(2:n, function(n_i) {
      if(n_i > length(y)) return(NULL)
      do.call(paste, unname(as.data.frame(embed(rev(y), n_i), stringsAsFactors=FALSE)), quote=FALSE)
    })))
  })
}

system.time({
  tokens <- stri_split_fixed(texts, ' ')
  tokens <- find_ngrams(tokens, n=2, verbose=TRUE)
  token_vector <- unlist(tokens)
  bagofwords <- unique(token_vector)
  n.ids <- sapply(tokens, length)
  i <- rep(seq_along(n.ids), n.ids)
  j <- match(token_vector, bagofwords)
  dt <- data.table(
    doc_id = i,
    ngram = bagofwords[j],
    cnt = 1L,
    key = c('doc_id', 'ngram')
  )
  dt <- dt[,list(cnt = sum(cnt)), by=c('doc_id', 'ngram')]
})
dt
