library(quanteda)
library(readr)
data(data_corpus_inaugural)
set.seed(42)
x <- data_corpus_inaugural$documents$texts
x <- c(x, x)
target <- sample(0:1, length(x), replace=T)
out <- data.table(
  `target` = target,
  `Inaugrual Speech 1` = x,
  `Inaugrual Speech 2` = sample(x)
)
write_csv(out, '~/datasets/dummy_text.csv')
