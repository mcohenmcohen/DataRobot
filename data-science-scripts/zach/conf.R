#https://www.kaggle.com/kobakhit/digit-recognizer/digital-recognizer-in-r
#COPY:

## Confusion Matrix - (vertical: actual; across: predicted): vertical: actual; across: predicted
##          0    1    2   3   4    5    6    7   8   9  Error           Rate
## 0      942    0    6   1   1    3    3    3   4   1 0.0228 =     22 / 964
## 1        0 1028    7   3   0    2    0    3   1   0 0.0153 =   16 / 1,044
## 2        3    2 1005   1   2    0    1    7   3   1 0.0195 =   20 / 1,025
## 3        0    0   30 952   0   12    1    9   4   0 0.0556 =   56 / 1,008
## 4        0    3    8   0 956    0    6    4   2  24 0.0469 =   47 / 1,003
## 5        0    2   16   6   4  975    7    3   1   2 0.0404 =   41 / 1,016
## 6        5    0   16   0   4    5  996    0   2   0 0.0311 =   32 / 1,028
## 7        1    3   23   1   1    1    0  993   0   9 0.0378 =   39 / 1,032
## 8        0   10   17   9   1    6    8    3 931   7 0.0615 =     61 / 992
## 9        4    1    5  11  11    4    0   21   7 880 0.0678 =     64 / 944
## Totals 955 1049 1133 984 980 1008 1022 1046 955 924 0.0396 = 398 / 10,056

library(stringi)
a <- read.delim(pipe('pbpaste'), stringsAsFactors = F)
row.names(a) <- names(a) <- NULL


b = do.call(rbind, stri_split_regex(a[,1], ' +'))[-1,-c(1:2)][1:10,1:10]
c = matrix(as.integer(b), ncol=10)

d = c
err1 = d / rowSums(d)
err2 = t(d) / rowSums(t(d))
err = err1 + err2
diag(err) <- 0
i = rep(FALSE, length(err))
i[which.max(as.numeric(err))] <- TRUE
i = matrix(i, ncol=10)

row.names(i) <- colnames(i) <- paste(0:9)
row.names(err) <- colnames(err) <- paste(0:9)

i
round(err, 2)



d = d + t(d)
err = rowSums(d)
tot = rowSums(c)

rt = err / (tot + rev(tot))
which.max(rt)
