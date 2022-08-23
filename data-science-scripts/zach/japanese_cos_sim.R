set.seed(42)

words_ja <- c(
  "あの人", "あのかた", "彼女", "彼", "です", "あります", "おります", "います",
  "は", "が", "の", "に", "を", "で", "え", "から", "まで", "より", "も",
  "どの", "と", "し", "それで", "しかし", "にんげん", "かんごし", "せんせい",
  "かぞく", "いぬ", 'ねこ', 'じかん', 'たいよう', 'つき'
)

words_en <- c(
  "apricot", "avocado", "banana", "Kiwifruit", "kiwifruit", "marionberry",
  "honeydew", "peach", "prune", "quince", "pomegranate", "Strawberry",
  "Tamarind", "orange", "apple", "organge", "peach", "plumb", "nectarine",
  "fruit", "vegtable", "tomato", "potato"
)

babble <- function(N = 25, vocab){
  sapply(1:N, function(x){
    stops <- sample(vocab, sample(10:25, 1), replace=TRUE)
    paste(stops, collapse=" ")
  })
}
make_data <- function(N, vocab){
  phrases <- babble(N/2, vocab)
  phrases_1 <- babble(N/2, vocab)
  phrases_2 <- babble(N/2, vocab)
  x_1 <- c(phrases, phrases_1)
  x_2 <- c(phrases, phrases_2)
  y <- as.integer(x_1 == x_2)
  out <- data.frame(y, x_1, x_2)
  out <- out[sample(1:nrow(out)),]
  return(out)
}

dat_ja <- make_data(500, words_ja)
dat_en <- make_data(500, words_en)

sum(dat_en$y)
sum(dat_ja$y)

write.csv(dat_ja, "~/datasets/japanese_cosine_sim.csv", row.names=FALSE)
write.csv(dat_en, "~/datasets/english_cosine_sim.csv", row.names=FALSE)
