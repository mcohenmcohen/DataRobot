
##################################################
# NAs
##################################################

set.seed(1)
library(caret)
#d <- twoClassSim(n=5700, intercept=-27)
d <- twoClassSim(n=5700, intercept=-53)
p <- 1/1:5
p <- p/sum(p)
words <- sapply(words, function(i){
  paste(sample(letters, sample(1:5, 1, prob=p), replace=TRUE), collapse='')
})
#d$text <- sapply(1:nrow(d), function(i){
  paste(sample(words, sample(1:5, 1, prob=p), replace=TRUE), collapse=' ')
})
yname <- 'Class'
others <- setdiff(names(d), yname)
d <- d[,c(yname, others)]
i <- which(d$Class == 'Class1')
table(d$Class, useNA='ifany')
write.csv(d, '~/datasets/only_one_positive_no_NA.csv')

d[sample(i, 1), 'Class'] <- NA
table(d$Class, useNA='ifany')
write.csv(d, '~/datasets/only_one_positive_1NAs.csv')

d[sample(i, 54), 'Class'] <- NA
table(d$Class, useNA='ifany')
write.csv(d, '~/datasets/only_one_positive.csv')


##################################################
# No NAs
##################################################

set.seed(1)
library(caret)
d <- twoClassSim(n=5700, intercept=-36.5)
yname <- 'Class'
others <- setdiff(names(d), yname)
d <- d[,c(yname, others)]
i <- which(d$Class == 'Class1')
d[sample(i, 26), 'Class'] <- NA
table(d$Class, useNA='ifany')
write.csv(d, '~/datasets/9_positive_26_NA.csv')
d <- d[!is.na(d$Class),]
write.csv(d, '~/datasets/9_positive_no_NA.csv')

##################################################
# Case 3
##################################################

set.seed(1)
library(caret)
d <- twoClassSim(n=5700, intercept=-35)
yname <- 'Class'
others <- setdiff(names(d), yname)
d <- d[,c(yname, others)]
i <- which(d$Class == 'Class1')
d$Class <- as.character(d$Class)
d[sample(i, 26), 'Class'] <- NA
table(d$Class, useNA='ifany')

write.csv(d, '~/datasets/9_case_3.csv')
