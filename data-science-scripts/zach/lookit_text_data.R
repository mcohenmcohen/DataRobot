library(data.table)
a=fread('https://s3.amazonaws.com/datarobot_public_datasets/bloggers_small_80.csv')
b=fread('https://s3.amazonaws.com/datarobot_public_datasets/ohsumed_binary_80.csv')
c=fread('https://s3.amazonaws.com/datarobot_public_datasets/reuters_earnVacq_80.csv')

#Bloggers
dim(a)
a[,table(gender)/.N]
a[,summary(nchar(post))]

#Ohsumed
dim(b)
b[,table(disease)/.N]
b[,summary(nchar(abstract))]

#Retuers
dim(c)
c[,table(subject)/.N]
c[,summary(nchar(article))]
