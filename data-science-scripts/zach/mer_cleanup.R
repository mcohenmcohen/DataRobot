library(data.table)
library(stringi)
#https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s

dat <- fread('~/datasets/mer_train.tsv')
dat <- dat[price > 0,]

dat[is.na(name), name := '']
dat[is.na(brand_name), brand_name := '']
dat[is.na(item_description), item_description := '']
dat[is.na(category_name), category_name := '']

dat[,item_condition_id := stri_c('c', item_condition_id)]
dat[,shipping := stri_c('s', shipping)]
dat[,name := stri_c(name, brand_name, sep = ' ')]
dat[,text := stri_c(item_description, name, category_name, sep = ' ')]

out <- dat[,list(target=price, name, text, item_condition_id, shipping)]
fwrite(out, '~/datasets/mer_text_combo.csv')
