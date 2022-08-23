x <- find_tokens(inaugTexts)
x <- c(x,x,x,x,x)
microbenchmark(
  t1={
    tokens <- lapply(x, wordStem)
  },
  t2={
    tokens <- as.relistable(x)
    tokens <- unlist(tokens)
    unique_tokens <- unique(tokens)
    token_map <- match(tokens, unique_tokens)
    unique_tokens <- wordStem(unique_tokens)
    tokens[] <- unique_tokens[token_map]
    tokens <- relist(tokens)
  },
  times=100)
