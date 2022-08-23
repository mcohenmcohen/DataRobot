# Setup
stop()
rm(list=ls(all=T))
gc(reset=T)
library(data.table)
library(pbapply)
library(stringi)
library(VGAM)

# Parameters
set.seed(42)
nrows <- 1e6
nwords <- 1000
vocabprob <- dzipf(1:nwords, nwords, shape=1.01)
vocabprob <- vocabprob/sum(vocabprob)
max_letters <- 7
letterprob <- max_letters:1/sum(max_letters:1)
max_words <- 20
wordprob <- (max_words:1)^2
wordprob <- wordprob/sum(wordprob)
round(wordprob, 3)

# Choose letters
letters_japanese <-
  c("あ", "ア", "い", "う", "え", "お", "か", "が", "ガ",
    "き", "キ", "ぎ", "く", "ク", "ぐ", "け", "げ", "こ",
    "ご", "さ", "ざ", "し", "じ", "す", "ス", "ず", "せ",
    "ぜ", "そ", "ぞ", "た", "だ", "ち", "っ", "つ", "づ",
    "て", "で", "と", "ど", "ド", "な", "に", "ぬ", "ね",
    "の", "ノ", "は", "ば", "ひ", "ふ", "ぶ", "へ", "べ",
    "ほ", "ぼ", "ま", "み", "む", "め", "も", "ゃ", "や",
    "ゆ", "ょ", "よ", "ら", "ラ", "り", "る", "れ", "ろ",
    "ロ", "わ", "ゐ", "ん", "ン", "一", "七", "三", "上",
    "下", "世", "中", "乗", "事", "二", "云", "五", "人",
    "今", "仕", "他", "以", "会", "体", "何", "作", "使",
    "供", "係", "信", "俺", "傍", "僕", "先", "光", "入",
    "八", "六", "内", "出", "分", "切", "初", "前", "剣",
    "力", "動", "十", "取", "受", "口", "合", "同", "名",
    "向", "君", "味", "呼", "唇", "問", "四", "国", "在",
    "地", "場", "壁", "声", "変", "外", "多", "夜", "大",
    "奥", "女", "奴", "好", "始", "姿", "娘", "子", "字",
    "存", "室", "家", "小", "少", "居", "屋", "山", "巳",
    "帰", "年", "床", "店", "度", "引", "強", "当", "形",
    "影", "彼", "待", "後", "得", "御", "心", "必", "忘",
    "思", "性", "息", "悪", "情", "意", "感", "戻", "所",
    "扉", "手", "持", "指", "振", "教", "数", "敵", "方",
    "日", "早", "明", "時", "書", "最", "本", "村", "来",
    "様", "横", "機", "次", "歩", "死", "残", "殺", "母",
    "気", "水", "活", "消", "深", "点", "父", "物", "王",
    "生", "男", "町", "界", "白", "百", "的", "目", "相",
    "眼", "知", "確", "祐", "神", "私", "空", "窓", "立",
    "竜", "笑", "第", "答", "終", "続", "緒", "線", "置",
    "考", "者", "耳", "聞", "肉", "肩", "背", "胸", "腕",
    "腰", "自", "船", "色", "若", "落", "葉", "血", "行",
    "表", "要", "見", "視", "言", "話", "説", "誰", "走",
    "足", "身", "車", "軍", "通", "道", "達", "違", "部",
    "配", "金", "長", "開", "間", "関", "離", "電", "音",
    "頃", "頭", "題", "顔", "風", "飛", "飲", "首", "馬",
    "驚", "高", "髪", "黒", "黙")

# Create text - letters or letters_japanese
# letters <- letters_japanese # Uncomment for Japanese
words <- sapply(1:nwords, function(x){
  paste(sample(letters, sample(1:max_letters, 1, prob=letterprob)), collapse='')
})
words[1] <- "MAGICWORD"
text <- pbsapply(1:nrows, function(x){
  paste(sample(words, sample(1:max_words, 1, prob=wordprob), prob=vocabprob), collapse=' ')
})
magic_word <- words[1]

# Make data
dat <- data.table(
  text,
  int1 = sample(c(T, F), nrows, replace=T)
)
dat[,has_magic_word := stri_detect_fixed(text, magic_word)]
dat[,target := xor(has_magic_word, int1)]
dat[,int1 := as.integer(int1)]
dat[,target := as.integer(target)]

# Tables
dat[,table(has_magic_word) / .N]
dat[,table(int1) / .N]
dat[,table(target) / .N]

dat[,table(has_magic_word, target) / .N]
dat[,table(int1, target) / .N]

dat[,table(has_magic_word, int1) / .N]
dat[,table(has_magic_word, int1, target) / .N]

# Remove leak
dat[,has_magic_word := NULL]

# Save file
idx <- runif(nrows) <= .80
fout <- '~/datasets/large_xor_text_eng.csv'
fwrite(dat[which( idx),], paste0(fout, '_80.csv'))
fwrite(dat[which(!idx),], paste0(fout, '_20.csv'))
