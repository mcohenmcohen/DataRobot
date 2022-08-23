data(iris)
words = mapply(sample, x=list(letters), size=sample(5:10, nrow(iris), replace=T))
words = sapply(words, paste, collapse=' ')
iris[['text']] = words
write.csv(iris, '~/datasets/iris_words.csv', row.names=F)
