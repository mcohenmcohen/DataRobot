#Round 1
library(stringi)
a <- read.csv('~/datasets/stopwords.csv')
a[a$remove.translation != '', ]
a <- a[a$remove.translation == '', ]
w <- stri_trim_both(a$word)
w <- sort(unique(w))
dput(w)

#Round 2
library(stringi)
b <- read.csv('~/datasets/stopwords-2.csv')
b <- b[b$source == 'Akira', ]
w2 <- stri_trim_both(b$word)
w2 <- sort(unique(w2))
dput(w2)

#Round 3
w3 <- c("・", "、", "！", "？", "…",  "。", "（", "）", "「", "」", "ー")
w3 <- expand.grid(w3, w3)
w3 <- paste(w3[,1], w3[,2])
dput(w3)
