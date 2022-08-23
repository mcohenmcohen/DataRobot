rm(list=ls(all=TRUE))
gc(reset=TRUE)
library(caret)
library(yaml)
library(pbapply)
library(data.table)
library(stringi)

#Some japanese characters
CHAR_DATA <-
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
#CHAR_DATA <- c(letters, " ")

set.seed(2926)
HICARD_CHARS <- CJ(
  sample(CHAR_DATA, 25),
  sample(CHAR_DATA, 25),
  sample(CHAR_DATA, 25))
HICARD_CHARS <- paste0(HICARD_CHARS$V1, HICARD_CHARS$V2, HICARD_CHARS$V3)
HICARD_CHARS <- sort(unique(HICARD_CHARS))

#Mislabel a probability vector
calcProbs <- function(x){
  print("=============================================")
  print('Calculating target prob')
  t1 <- system.time({
    prob <- binomial()$linkinv(x)
  })
  print(t1)
  print("--------------------------------------------")
  return(prob)
}

#Mislabel a probability vector
mislabelProbs <- function(prob, mislabel){
  if (mislabel > 0 & mislabel < 1) {
    print("=============================================")
    print('Shuffling some classes')
    t1 <- system.time({
      shuffle <- sample(1:length(prob), floor(length(prob) * mislabel))
      prob[shuffle] <- 1 - prob[shuffle]
    })
    print(t1)
    print("--------------------------------------------")
  }
  return(prob)
}

#Turn probs into classes
calClasses <- function(prob){
  n = length(prob)
  print("=============================================")
  print('Calculating Classes')
  t1 <- system.time({
    Class <- paste0('Class', round(prob <= runif(n)) + 1)
  })
  print(t1)
  print("--------------------------------------------")
  return(Class)
}

#Add Missing data to a data.table
addMissing <- function(x, pct_missing){
  if(pct_missing > 0 & pct_missing < 1){
    print("=============================================")
    print('Adding missing data to X and Y')
    t1 <- system.time({
      for(col in colnames(x)){
        na <- sample(1:nrow(x), nrow(x) * pct_missing)
        set(x, i=na, j=col, value=NA)
      }
    })
    print(t1)
    print("--------------------------------------------")
  }
  return(x)
}

#Add low cardinality categoricals to a data.table
addLowCardNoise <- function(dat, n_low_card){
  if(n_low_card > 0){
    print("=============================================")
    print('Adding low card irrelevant')
    t1 <- system.time({
      for(i in 1:n_low_card){
        n <- 1:sample(2:5, 1)
        levels <- 1:length(n)
        labels <- CHAR_DATA[n]
        x <- sample(1:length(n), nrow(dat), replace=TRUE)
        x <- factor(x, levels=levels, labels=labels)
        set(dat, j=paste0('NoiseLowCard', i), value=x)
      }
    })
    print(t1)
    print("--------------------------------------------")
  }
  return(dat)
}

#Add high cardinality categoricals to a data.table
addHighCardNoise <- function(dat, n_high_card){
  if(n_high_card > 0){
    print("=============================================")
    print('Adding high card irrelevant')
    t1 <- system.time({
      for(i in 1:n_high_card){
        n <- 1:sample(100:10000, 1)
        levels <- 1:length(n)
        labels <- HICARD_CHARS[n]
        x <- sample(1:length(n), nrow(dat), replace=TRUE)
        x <- factor(x, levels=levels, labels=labels)
        set(dat, j=paste0('NoiseHighCard', i), value=x)
      }
    })
    print(t1)
    print("--------------------------------------------")
  }
  return(dat)
}

#Add numeric noise
#caret:::make_noise
makeNumericNoise <- function(
  n,
  noiseVars = 0,
  corrVars = 0,
  corrValue = 0,
  binaryNoise = FALSE)
{

  if (noiseVars > 0) {
    print("=============================================")
    print('Starting gaussian noise vars')
    t1 <- system.time({
      noise <-  matrix(rnorm(n * noiseVars), ncol = noiseVars)
      colnames(noise) <- paste0("Noise", 1:ncol(noise))
    })
    print(t1)
    print("--------------------------------------------")
  } else{
    noise <- NULL
  }

  if (corrVars > 0) {
    print("=============================================")
    print('Starting correlated gaussian noise vars')
    t1 <- system.time({
      vc <- matrix(corrValue, ncol = corrVars, nrow = corrVars)
      diag(vc) <- 1
      cor <- MASS::mvrnorm(n, mu = rep(0, corrVars), Sigma = vc)
      colnames(cor) <- paste0('NoiseCorr', 1:ncol(cor))
    })
    print(t1)
    print("--------------------------------------------")
  } else{
    cor <- NULL
  }

  out <- cbind(noise, cor)

  if(binaryNoise){
    out <- round(out)
    out <- pmax(out, 0)
    out <- pmin(out, 1)
  }

  return(out)
}

##################################################################
#Generate classification dataset 1
##################################################################
#caret::twoClassSim
#Bigger data version of twoClassSim:
twoClassSimBig <- function(
  n,
  intercept = -25,
  linearVars = 20,
  mislabel = .01,
  noiseVars = 10,
  corrVars = 10,
  corrValue = 0.25,
  binaryNoise = FALSE,
  n_low_card = 5,
  n_high_card = 5,
  pct_missing = .05) {

  print("=============================================")
  print('Starting two factor vars')
  t1 <- system.time({
    sigma <- matrix(c(2, 1.3, 1.3, 2), 2, 2)
    twoClass <- MASS::mvrnorm(n = n, c(0, 0), sigma)
    colnames(twoClass) <- paste("TwoFactor", 1:2, sep = "")
  })
  print(t1)
  print("--------------------------------------------")

  if (linearVars > 0) {
    print("=============================================")
    print('Starting linear vars')
    t1 <- system.time({
      lin <- matrix(
        rnorm(n * linearVars),
        ncol = linearVars)
      colnames(lin) <- paste0("Linear", 1:ncol(lin))
    })
    print(t1)
    print("--------------------------------------------")
  }

  print("=============================================")
  print('Starting non-linear vars')
  t1 <- system.time({
    nonlin1 <- runif(n, min = -1)
    nonlin2 <- matrix(runif(n * 2), ncol = 2)
    nonlin <- cbind(nonlin1, nonlin2)
    rm(nonlin1, nonlin2)
    colnames(nonlin) <- paste0('Nonlinear', 1:ncol(nonlin))
  })
  print(t1)
  print("--------------------------------------------")

  print("=============================================")
  print('Calculating linear target')
  t1 <- system.time({
    lp <- intercept -
      4 * twoClass[,'TwoFactor1'] +
      4 * twoClass[,'TwoFactor2'] +
      2 * twoClass[,'TwoFactor1'] * twoClass[,'TwoFactor2'] +
      (nonlin[,'Nonlinear1']^3) +
      2 * exp(-6 * (nonlin[,'Nonlinear1'] - 0.3)^2) +
      2 * sin(pi * nonlin[,'Nonlinear2'] * nonlin[,'Nonlinear3'])
  })
  print(t1)
  print("--------------------------------------------")

  if (linearVars > 0) {
    print("=============================================")
    print("Adding linear coefs")
    t1 <- system.time({
      cf <- seq(10, 1, length = linearVars)/4
      cf <- cf * rep(c(-1, 1), floor(linearVars) + 1)[1:linearVars]
      lp <- lp + lin %*% cf
    })
    print(t1)
    print("--------------------------------------------")
  }

  prob <- calcProbs(lp)
  prob <- mislabelProbs(prob, mislabel)
  Class <- calClasses(prob)

  noise <- makeNumericNoise(
    n,  noiseVars = noiseVars,
    corrVars = corrVars,
    corrValue = corrValue,
    binaryNoise=binaryNoise)

  print("=============================================")
  print('Assembling final dataset')
  t1 <- system.time({
    dat <- data.table(
      Class=Class,
      twoClass,
      lin,
      nonlin,
      noise
    )
  })
  print(t1)
  print("--------------------------------------------")

  dat <- addMissing(dat, pct_missing=pct_missing)
  dat <- addLowCardNoise(dat, n_low_card)
  dat <- addHighCardNoise(dat, n_high_card)

  return(dat)
}

if(FALSE){
  set.seed(6872)
  n <- 1.5e7
  system.time(dat <- twoClassSimBig(n))
  print(object.size(dat), units='Gb')
  fname <- paste0('~/datasets/dirty_simulated_V2_', n)
  write.csv(dat, fname)
  s <- file.size(fname)
  print(1e10/s * n)
  print(gdata::humanReadable(s))
  table(dat$Class) / nrow(dat)
  table(dat$Class) / nrow(dat[!is.na(dat$Class),])
}

##################################################################
#Generate classification dataset 2
##################################################################
#caret::LPH07_1
LPH07_1_Big <- function(
  n = 100,
  linearVars = 10,
  mislabel = .10,
  noiseVars = 10,
  corrVars = 10,
  corrValue = 0.10,
  binaryNoise = TRUE,
  n_low_card = 10,
  n_high_card = 10,
  pct_missing = .10,
  class = TRUE){

  w <- matrix(rbinom(n * 10, size = 1, prob = 0.4), ncol = 10)
  colnames(w) <- paste0("Var", 1:ncol(w))

  y <- 2 * w[,1] * w[,10] + 4 * w[,2] * w[,7] + 3 *
    w[,4] * w[,5] - 5 * w[,6] * w[10] + 3 * w[,8] * w[,9] + w[,1] *
    w[,2] * w[,4] - 2 * w[,7] * (1 - w[,6]) * w[,2] * w[,9] - 4 *
    (1 - w[,10]) * w[,1] * (1 - w[,4])

  if (class) {
    y <- calcProbs(y)
    y <- mislabelProbs(y, mislabel)
    y <- calClasses(y)
  } else {
    y <- y + rnorm(n)
  }

  noise <- makeNumericNoise(
    n,  noiseVars = noiseVars,
    corrVars = corrVars,
    corrValue = corrValue,
    binaryNoise=binaryNoise)

  dat <- data.table(target=y, w, noise)

  dat <- addMissing(dat, pct_missing=pct_missing)
  dat <- addLowCardNoise(dat, n_low_card)
  dat <- addHighCardNoise(dat, n_high_card)
  return(dat)
}

if(FALSE){
  set.seed(3824)
  n <- 4e7
  system.time(dat <- LPH07_1_Big(n))
  print(object.size(dat), units='Gb')
  fname <- paste0('~/datasets/LPH07_dirty_', n)
  write.csv(dat, fname)
  s <- file.size(fname)
  print(1e10/s * n)
  print(gdata::humanReadable(s))
  table(dat$target) / nrow(dat)
}

##################################################################
#Generate regression dataset 1
##################################################################
#caret::SLC14_1
SLC14_1_Big <- function(
  n = 100,
  linearVars = 10,
  mislabel = .10,
  noiseVars = 10,
  corrVars = 10,
  corrValue = 0.10,
  binaryNoise = FALSE,
  n_low_card = 10,
  n_high_card = 10,
  pct_missing = .10
){

  x <- matrix(rnorm(n * 20, sd = 3), ncol = 20)

  y <- x[,1] +
    sin(x[,2]) +
    log(abs(x[,3])) +
    x[,4]^2 +
    x[,5] * x[,6] +
    ifelse(x[,7] * x[,8] * x[,9] < 0, 1, 0) +
    ifelse(x[,10] > 0, 1, 0) +
    x[,11] * ifelse(x[,11] > 0, 1, 0) +
    sqrt(abs(x[,12])) +
    cos(x[,13]) +
    2 * x[,14] +
    abs(x[,15]) +
    ifelse(x[,16] < -1, 1, 0) +
    x[,17] * ifelse(x[,17] < -1, 1, 0) -
    2 * x[,18] -
    x[,19] * x[,20] +
    rnorm(n, sd = 3) +
    rpois(n, 1)

  colnames(x) <- paste0('Var', 1:ncol(x))

  noise <- makeNumericNoise(
    n,  noiseVars = noiseVars,
    corrVars = corrVars,
    corrValue = corrValue,
    binaryNoise = binaryNoise)

  dat <- data.table(target=y, x, noise)

  dat <- addMissing(dat, pct_missing=pct_missing)
  dat <- addLowCardNoise(dat, n_low_card)
  dat <- addHighCardNoise(dat, n_high_card)
  return(dat)
}

if(FALSE){
  set.seed(87275)
  n <- 1.3e7
  system.time(dat <- SLC14_1_Big(n))
  print(object.size(dat), units='Gb')
  fname <- paste0('~/datasets/SLC14_1_dirty_', n)
  write.csv(dat, fname)
  s <- file.size(fname)
  print(1e10/s * n)
  print(gdata::humanReadable(s))
  summary(dat$target)
}

##################################################################
#Generate regression dataset 2 - Many columns
##################################################################
#caret::SLC14_2
SLC14_2_Big <- function(
  n = 100,
  num_cols = 200,
  mislabel = .10,
  noiseVars = 10,
  corrVars = 10,
  corrValue = 0.10,
  binaryNoise = TRUE,
  n_low_card = 10,
  n_high_card = 10,
  pct_missing = .10,
  class=FALSE,
  intercept=-10
){

  x <- matrix(rnorm(n * num_cols, sd = 4), ncol = num_cols)
  y <- rowSums(log(abs(x))) +
    rnorm(n, sd = 5) - 1 + rpois(n, 5)
  colnames(x) <- paste0('Var', 1:ncol(x))

  noise <- makeNumericNoise(
    n,  noiseVars = noiseVars,
    corrVars = corrVars,
    corrValue = corrValue,
    binaryNoise = binaryNoise)

  if (class) {
    y <- y - mean(y)
    y <- y - intercept
    y <- calcProbs(y)
    y <- mislabelProbs(y, mislabel)
    y <- calClasses(y)
  }

  dat <- data.table(target=y, x, noise)

  dat <- addMissing(dat, pct_missing=pct_missing)
  dat <- addLowCardNoise(dat, n_low_card)
  dat <- addHighCardNoise(dat, n_high_card)
  return(dat)
}

if(FALSE){
  set.seed(59632)
  n <- 3e6
  system.time(dat <- SLC14_2_Big(n))
  print(object.size(dat), units='Gb')
  fname <- paste0('~/datasets/SLC14_2_dirty_', n)
  write.csv(dat, fname)
  s <- file.size(fname)
  print(1e10/s * n)
  print(gdata::humanReadable(s))
  summary(dat$target)
}

if(FALSE){
  # Special case for bug testing on 2016-03-23
  # Using CHAR_DATA <- c(letters, " ")
  set.seed(25576)
  n <- as.integer(floor(110521*1.8))
  system.time(dat <- SLC14_2_Big(
    n,
    num_cols=500,
    mislabel=0,
    noiseVars=100,
    corrVars=50,
    n_low_card=25,
    n_high_card=25,
    class=TRUE,
    intercept=-35
  ))
  dat <- dat[!is.na(target),]
  print(object.size(dat), units='Mb')
  dim(dat)
  table(dat$target)
  table(dat$target) / nrow(dat)

  fname <- paste0('~/datasets/SLC14_2_dirty_class_english', n, ".csv")
  write.csv(dat, fname)
  s <- file.size(fname)
  print(gdata::humanReadable(s))
  print(1e10/s * n)
}

##################################################################
#Generate regression dataset 3
##################################################################
#caret::LPH07_2
LPH07_2_Big <- function(
  n = 100,
  linearVars = 10,
  mislabel = .10,
  noiseVars = 10,
  corrVars = 10,
  corrValue = 0.10,
  binaryNoise = FALSE,
  n_low_card = 10,
  n_high_card = 10,
  pct_missing = .10
){

  x <- matrix(rnorm(n * 20, sd = 4), ncol = 20)
  colnames(x) <- paste0('Var', 1:ncol(x))
  y <- x[,1] * x[,2] + x[,10]^2 - x[,3] * x[,17] -
    x[,15] * x[,4] + x[,9] * x[,5] + x[,19] - x[,20]^2 + x[,9] *
    x[,8] + rnorm(n, sd = 4)

  noise <- makeNumericNoise(
    n,  noiseVars = noiseVars,
    corrVars = corrVars,
    corrValue = corrValue,
    binaryNoise = binaryNoise)

  dat <- data.table(target=y, x, noise)

  dat <- addMissing(dat, pct_missing=pct_missing)
  dat <- addLowCardNoise(dat, n_low_card)
  dat <- addHighCardNoise(dat, n_high_card)
  return(dat)
}

if(FALSE){
  set.seed(5170)
  n <- 1.3e7
  system.time(dat <- LPH07_2_Big(n))
  print(object.size(dat), units='Gb')
  fname <- paste0('~/datasets/LPH07_2_dirty_', n)
  write.csv(dat, fname)
  s <- file.size(fname)
  print(1e10/s * n)
  print(gdata::humanReadable(s))
  summary(dat$target)
}
