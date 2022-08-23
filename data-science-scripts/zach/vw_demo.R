library(data.table)
library(stringi)
library(caTools)
dat <- fread('~/Downloads/item5.csv')
dat[score==0, score := -1]
dat[, response := stri_replace_all_fixed(response, ':', ' ')]
dat[, response := stri_replace_all_fixed(response, '|', ' ')]
dat[, response := stri_replace_all_fixed(response, '\n', ' ')]
dat[,response := tolower(response)]
dat[,vw := paste0(score, ' |response ', response)]
head(dat$vw)

dat <- dat[order(runif(.N)),]
head(dat$vw)

cat(dat[1:1000,vw], file='vw.train.txt', sep='\n')
cat(dat[1001:1468,vw], file='vw.test.txt', sep='\n')
system('head vw.train.txt')
system('head vw.test.txt')

system('vw vw.train.txt -c --passes 100 --loss_function hinge -f model.vw --learning_rate .10 --ngram 2 --ngram 3 --skips 2 --l2 1e-8')

system('vw vw.test.txt --testonly -i model.vw -p pred.txt --link logistic --invert_hash readable.txt')
c <- fread('readable.txt')
p <- fread('pred.txt')

colAUC(p$V1, dat[1001:1468,score])



