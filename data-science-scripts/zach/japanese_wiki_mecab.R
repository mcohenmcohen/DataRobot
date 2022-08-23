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
#https://dumps.wikimedia.org/jawiki/latest/
#Retrieved rarly June 2016
#system("./WikiExtractor.py ~/datasets/jawiki-latest-pages-articles.xml.bz2 -o ~/datasets/jawiki --no-templates -ns Main --no-doc --no-title --processes 9")

#NEW ONE (not yet run):
#system("./WikiExtractor.py ~/datasets/jawiki-latest-pages-articles.xml.bz2 -o ~/datasets/jawiki --no-templates --xml_namespaces 0 --no-doc --no-title --processes 9 --min_text_length 50  --filter_disambig_pages")

#Make list of articles
dirs <- list.files("~/datasets/jawiki", full.names = T)
fls <- unname(unlist(sapply(dirs, function(x) paste0(x, '/', list.files(x)))))

#Count articles
#http://unix.stackexchange.com/a/82958
#http://unix.stackexchange.com/a/4482
count_lines <- function(f){
  as.integer(system(paste("grep -cvE '[^[:space:]]'", f), intern=TRUE))
}
articles <- pbsapply(fls, count_lines)
total_articles <- sum(articles, na.rm=TRUE)

#Mecab the articles
run_mecab <- function(file_in, overwrite=FALSE){
  file_out <- gsub("jawiki", "jawiki_mecab", file_in, fixed=TRUE)
  cmd <- paste0(
    "mecab -O wakati ", file_in,
    " -o", file_out
  )
  if((!file.exists(file_out)) | overwrite){
    unlink(file_out)
    system(cmd)
  }
}

sink <- pblapply(fls, run_mecab)

#Make list of mecab'd articles
mecab_dirs <- list.files("~/datasets/jawiki_mecab", full.names = T)
mecab_fls <- unname(unlist(sapply(mecab_dirs, function(x) paste0(x, '/', list.files(x)))))

#Define reader and tokenizer for mecab docs
reader <- function(x){
  out <- read_file(x)
  out <- stri_split_fixed(out, '\n\n')[[1]]
  out <- stri_replace_all_regex(out, '[\\p{C}|\\p{M}|\\p{N}|\\p{P}|\\p{S}|\\p{Z}]+', ' ')
  out <- stri_replace_all_regex(out, ' +', ' ')
  out <- stri_trim_both(out)
  out <- out[nchar(out) > 100]
  return(out)
}
tokenizer <- function(x) stri_split_fixed(x, " ", fixed=TRUE)
read_and_tokenize <- function(x){itoken(ifiles(x, reader_function = reader), tokenizer = tokenizer)}

#Create vocabulary - 2grams
#http://dsnotes.com/articles/text2vec-0-3
#http://dsnotes.com/articles/glove-enwiki
v <- create_vocabulary(read_and_tokenize(mecab_fls), ngram=c(ngram_min = 1L, ngram_max = 2L))
v_prune <- prune_vocabulary(v, doc_proportion_min=.10)$vocab
v_prune[,terms := stri_replace_all_fixed(terms, '_', ' ')]
v_prune <- v_prune[terms != '',]
setorderv(v_prune, 'doc_counts')
saveRDS(v_prune, file='~/datasets/japanese_mecab_stopwords_2gram.RDS')

#Add some words
extras <- c(
  "あそこ", "あたり", "あちら", "あっち", "あと",
  "あな", "あなた", "あの", "あのかた", "あの人",
  "あります", "ある", "あれ", "いくつ", "いつ", "いま",
  "います", "いや", "いろいろ", "うち", "え", "おおまか",
  "おまえ", "おります", "おれ", "が", "がい", "かく",
  "かたち", "かやの", "から", "がら", "ヵ所", "カ所",
  "ヵ月", "カ月", "きた", "くせ", "くれ", "ヶ所", "ヶ月",
  "ここ", "こちら", "こっち", "ごっちゃ", "こと",
  "ごと", "この", "これ", "これら", "ごろ", "さまざま",
  "さらい", "さん", "し", "しかし", "しかた", "しょ",
  "しよう", "すか", "ずつ", "すね", "すべて", "する",
  "ぜんぶ", "そう", "そこ", "そちら", "そっち", "そで",
  "その", "その後", "それ", "それぞれ", "それで",
  "それなり", "たくさん", "だけ", "たし", "たち",
  "たび", "ため", "だめ", "だれ", "だろ", "ちゃ", "ちゃん",
  "で", "です", "てる", "てん", "と", "とおり", "とか",
  "とき", "どこ", "どこか", "ところ", "どちら", "どっか",
  "どっち", "とて", "どの", "どれ", "ない", "なか",
  "なかっ", "なかば", "など", "なに", "なら", "なん",
  "に", "にかく", "の", "は", "ハイ", "はじめ", "はず",
  "はるか", "ひと", "ひとつ", "ふく", "ぶり", "べき",
  "べつ", "へん", "ぺん", "ほう", "ほか", "まさ", "まし",
  "ませ", "まで", "まとも", "まま", "みたい", "みつ",
  "みなさん", "みんな", "も", "もう", "もと", "もの",
  "もん", "やつ", "よう", "よそ", "より", "レ", "わけ",
  "わたし", "を", "一", "一つ", "七", "万", "三", "上",
  "上記", "下", "下記", "中", "九", "事", "二", "五",
  "人", "今", "今回", "他", "以上", "以下", "以前",
  "以後", "以降", "会", "伸", "体", "何", "何人", "作",
  "例", "係", "俺", "個", "億", "元", "兆", "先", "全部",
  "八", "六", "内", "円", "冬", "分", "列", "別", "前",
  "前回", "力", "化", "匹", "区", "十", "千", "半ば",
  "口", "台", "右", "各", "同じ", "名", "名前", "向こう",
  "哀", "品", "員", "喜", "器", "四", "回", "国", "土",
  "地", "場", "場合", "境", "士", "夏", "外", "多く",
  "女", "奴", "婦", "子", "字", "室", "家", "屋", "左",
  "市", "席", "年", "年生", "幾つ", "店", "府", "度",
  "式", "形", "彼", "彼女", "後", "怒", "性", "情", "感",
  "感じ", "我々", "所", "手", "手段", "扱い", "数",
  "文", "新た", "方", "方法", "日", "春", "時", "時点",
  "時間", "書", "月", "木", "未満", "本当", "村", "束",
  "枚", "校", "楽", "様", "様々", "次", "歳", "歴", "段",
  "毎", "毎日", "気", "水", "法", "火", "点", "玉", "用",
  "男", "町", "界", "略", "百", "的", "目", "県", "確か",
  "私", "私達", "秋", "秒", "第", "等", "箇所", "箇月",
  "簿", "系", "紀", "結局", "線", "者", "自体", "自分",
  "行", "見", "観", "話", "誌", "誰", "課", "論", "貴方",
  "貴方方", "輪", "近く", "通", "連", "週", "道", "達",
  "違い", "部", "都", "金", "間", "関係", "際", "集",
  "面", "頃", "類", "首", "高")

#Remove some words
mystops <- sort(unique(c(v_prune$terms, extras)))
remove <- c("日本", "存在", "活動", "世界")
mystops <- setdiff(mystops, remove)

#Format for python
length(mystops)
out <- paste0('frozenset(["', paste(mystops, collapse='", "'), '"])')
cat(out)

#Create vocabulary - unigrams
#http://dsnotes.com/articles/text2vec-0-3
#http://dsnotes.com/articles/glove-enwiki
v1 <- create_vocabulary(read_and_tokenize(mecab_fls), ngram=c(ngram_min = 1L, ngram_max = 1L))
v_prune <- prune_vocabulary(v1, doc_proportion_min=.10)$vocab
v_prune[,terms := stri_replace_all_fixed(terms, '_', ' ')]
v_prune <- v_prune[terms != '',]
setorderv(v_prune, 'doc_counts')
saveRDS(v_prune, file='~/datasets/japanese_mecab_stopwords_1gram.RDS')

#Format for python
out <- paste0('frozenset(["', paste(v_prune$terms, collapse='", "'), '"])')
cat(out)
