library(data.table)
library(stringi)
library(textutils)

more_than_100.csv = fread('~/Downloads/more_than_100.csv')
chars = unlist(lapply(more_than_100.csv, stri_unique))
chars = stri_unique(chars)
chars = strsplit(chars, '', fixed=T)
chars = unlist(lapply(chars, stri_unique))
chars = sort(stri_unique(chars))
chars[stringi::stri_enc_mark(chars) != 'ASCII']
URLencode(chars[stringi::stri_enc_mark(chars) != 'ASCII'][1])
URLencode(chars[stringi::stri_enc_mark(chars) != 'ASCII'][2])
