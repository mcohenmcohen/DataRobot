set.seed(42)

stops_ja <- c(
  "これ", "それ", "あれ", "この", "その", "あの", "ここ", "そこ", "あそこ", "こちら",
  "どこ", "だれ", "なに", "なん", "何", "私", "貴方", "貴方方", "我々", "私達",
  "あの人", "あのかた", "彼女", "彼", "です", "あります", "おります", "います", "は",
  "が", "の", "に", "を", "で", "え", "から", "まで", "より", "も", "どの", "と",
  "し", "それで", "しかし"
)
stops_en <- c(
  "all", "show", "anyway", "four", "latter", "go", "mill", "find", "seemed",
  "whose", "nevertheless", "everything", "with", "whoever", "enough", "should",
  "to", "only", "whom", "under", "do", "his", "get", "very", "de", "none",
  "cannot", "every", "during", "him", "is", "did", "cry", "this", "she",
  "where", "ten", "up", "namely", "are", "further", "even", "what", "please",
  "its", "behind", "above", "between", "it", "neither", "ever", "across", "can",
  "we", "full", "never", "however", "here", "others", "hers", "along",
  "fifteen", "both", "last", "many", "whereafter", "wherever", "against",
  "etc", "s", "became", "whole", "otherwise", "among", "via", "co",
  "afterwards", "had", "whatever", "alone", "moreover", "throughout",
  "yourself", "from", "would", "two", "been", "next", "eleven", "much", "call",
  "therefore", "interest", "themselves", "thr", "until", "empty", "more",
  "fire", "latterly", "hereby", "else", "everywhere", "former", "those",
  "must", "me", "myself", "these", "bill", "will", "while", "anywhere", "nine",
  "thin", "theirs", "my", "give", "almost", "sincere", "thus", "herein", "cant",
  "itself", "something", "in", "ie", "if", "perhaps", "six", "amount", "same",
  "wherein", "beside", "how", "several", "may", "after", "upon", "hereupon",
  "such", "a", "off", "whereby", "third", "together", "whenever", "well",
  "rather", "without", "so", "the", "con", "yours", "just", "less", "being",
  "indeed", "over", "move", "front", "own", "through", "yourselves", "fify",
  "still", "yet", "before", "thence", "somewhere", "thick", "seems", "except",
  "ours", "has", "might", "into", "then", "them", "someone", "around",
  "thereby", "five", "they", "not", "amp", "now", "nor", "name", "hereafter",
  "always", "whither", "either", "each", "become", "side", "therein", "twelve",
  "because", "often", "doing", "eg", "some", "back", "our", "beyond",
  "ourselves", "out", "for", "bottom", "although", "since", "forty", "per",
  "re", "does", "am", "three", "thereupon", "be", "whereupon", "nowhere",
  "besides", "found", "sixty", "anyhow", "by", "on", "about", "anything", "of",
  "could", "put", "keep", "whence", "due", "ltd", "hence", "onto", "or",
  "first", "already", "seeming", "formerly", "thereafter", "within", "one",
  "down", "everyone", "done", "another", "couldnt", "your", "fill", "her",
  "few", "twenty", "top", "there", "system", "least", "t", "anyone", "their",
  "too", "hundred", "was", "himself", "elsewhere", "mostly", "that", "becoming",
  "nobody", "but", "somehow", "part", "herself", "than", "he", "made",
  "whether", "see", "us", "i", "below", "un", "were", "toward", "and",
  "describe", "beforehand", "mine", "an", "meanwhile", "as", "sometime",
  "at", "have", "seem", "any", "inc", "again", "hasnt", "no", "whereas", "when",
  "detail", "also", "other", "take", "which", "becomes", "yo", "towards",
  "though", "who", "most", "eight", "amongst", "nothing", "why", "don", "noone",
  "sometimes", "amoungst", "serious", "having", "once"
)
stops_fr <- c(
  "eûtes", "êtes", "fussiez", "aurions", "serait", "le", "serais", "mais", "la",
  "eue", "tu", "ayante", "eux", "aux", "te", "eus", "ta", "aviez", "de", "fut",
  "fûtes", "moi", "sont", "mon", "ayant", "serez", "du", "nos", "aurez",
  "eussiez", "qu", "d", "furent", "fût", "étée", "soit", "leur", "t", "étés",
  "seriez", "en", "ses", "fus", "avons", "l", "eu", "et", "sommes", "aurais",
  "aurait", "es", "est", "eurent", "serions", "sur", "lui", "soyons", "ayants",
  "étais", "soyez", "que", "mes", "qui", "je", "même", "à", "c", "ayons", "s",
  "eûmes", "une", "ou", "était", "été", "étants", "étées", "ce", "son",
  "auriez", "avais", "étante", "ont", "suis", "fûmes", "avait", "avec",
  "fussions", "seraient", "avez", "eussions", "toi", "ton", "eues", "vous",
  "étaient", "aies", "on", "auront", "aurons", "avions", "eut", "me", "ayantes",
  "ma", "auras", "fussent", "ait", "des", "dans", "pour", "n", "ces", "seras",
  "un", "serai", "sera", "aie", "ayez", "avaient", "aurai", "votre", "étiez",
  "ai", "m", "j", "eussent", "eusses", "étantes", "auraient", "as", "au", "il",
  "sois", "vos", "étions", "par", "tes", "fusses", "aient", "ne", "étant",
  "seront", "serons", "aura", "eût", "notre", "elle", "pas", "nous", "fusse",
  "eusse", "y", "soient", "sa", "se")

make_stops <- function(
  N = 250, labels = c("bad", "ok", "good"), stops = stops_en, name='en'
){

  x <- sample(levels, N, replace=TRUE)
  y <- x + rnorm(N) / 2
  levels <- c(-1, 0, 1)

  x <- factor(x, levels=levels, labels=labels)
  stops_start <- sapply(1:N, function(x){
    stops <- sample(stops, sample(10:25, 1), replace=TRUE)
    paste(stops, collapse=" ")
  })
  stops_end <- sapply(1:N, function(x){
    stops <- sample(stops, sample(10:25, 1), replace=TRUE)
    paste(stops, collapse=" ")
  })

  x <- paste(stops_start, x, stops_end)

  out <- data.frame(y, x)
  names(out)[1] = name
  names(out)[2] = paste0("x_", name)
  return(out)
}

N <- 500
dat_ja <- make_stops(500, c("悪い", "平凡", "良い"), stops_ja, 'ja')
dat_en <- make_stops(500, c("bad", "ok", "good"), stops_en, 'en')
dat_fr <- make_stops(500, c("mal", "médiocre", "bien"), stops_fr, 'fr')

dat <- data.frame(dat_ja, dat_en, dat_fr)
y <- dat$ja + dat$en + dat$fr + runif(N) / 10
dat <- data.frame(y, dat[,c('x_ja', 'x_en', 'x_fr')])

write.csv(dat_ja, "~/datasets/japanese_stops.csv", row.names=FALSE)
write.csv(dat, "~/datasets/multilingual_stops.csv", row.names=FALSE)
