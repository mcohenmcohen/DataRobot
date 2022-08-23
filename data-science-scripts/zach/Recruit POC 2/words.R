system.time({
  set.seed(1)
  samplefun <- function(n, x, collapse){
    paste(sample(x, n, replace=TRUE), collapse=collapse)
  }
  words <- sapply(rpois(10000, 3) + 1, samplefun, letters, '')
  sents1 <- sapply(rpois(1000000, 5) + 1, samplefun, words, ' ')
  sents2 <- sapply(rpois(100000, 500) + 1, samplefun, words, ' ')
})

find_ngrams <- function(dat, n, verbose=FALSE){
  library(pbapply)
  stopifnot(is.list(dat))
  stopifnot(is.numeric(n))
  stopifnot(n>0)
  if(n == 1) return(dat)
  pblapply(dat, function(y) {
    if(length(y)<=1) return(y)
    c(y, unlist(lapply(2:n, function(n_i) {
      if(n_i > length(y)) return(NULL)
      do.call(paste, unname(as.data.frame(embed(rev(y), n_i), stringsAsFactors=FALSE)), quote=FALSE)
    })))
  })
}

text_to_ngrams <- function(sents, n=2){
  library(stringi)
  library(Matrix)
  tokens <- stri_split_fixed(sents, ' ')
  tokens <- find_ngrams(tokens, n=n, verbose=TRUE)
  token_vector <- unlist(tokens)
  bagofwords <- unique(token_vector)
  n.ids <- sapply(tokens, length)
  i <- rep(seq_along(n.ids), n.ids)
  j <- match(token_vector, bagofwords)
  M <- sparseMatrix(i=i, j=j, x=1L)
  colnames(M) <- bagofwords
  return(M)
}

rm(zach_ng2, tau_ng2); gc(reset=TRUE)
zach_t1 <- system.time(zach_ng1 <- text_to_ngrams(sents1))
tau_t1 <- system.time(tau_ng1 <- tau::textcnt(as.list(sents1), n = 2L, method = "string", recursive = TRUE))
tau_t1 / zach_t1

rm(zach_ng1, tau_ng1); gc(reset=TRUE)
zach_t2 <- system.time(zach_ng2 <- text_to_ngrams(sents2))
tau_t2 <- system.time(tau_ng2 <- tau::textcnt(as.list(sents2), n = 2L, method = "string", recursive = TRUE))
tau_t2 / zach_t2
