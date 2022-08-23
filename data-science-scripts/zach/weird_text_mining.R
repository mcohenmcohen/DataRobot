library(data.table)
library(stringi)
library(stringdist)
dat = fread('~/workspace/data-science-scripts/zach/lc900_clean.csv')
cf = fread('~/workspace/data-science-scripts/zach/desc_auto_tuned_text_cf.txt')

bad_line <- dat[362,desc]
print(bad_line)

bad_line <- stri_replace_all_fixed(bad_line, '$', '')
bad_line <- stri_replace_all_fixed(bad_line, '%', '')
bad_line <- stri_replace_all_fixed(bad_line, '.', '')
bad_line <- stri_replace_all_fixed(bad_line, '(', '')
bad_line <- stri_replace_all_fixed(bad_line, ')', '')
bad_line <- stri_replace_all_fixed(bad_line, '&', '')
bad_line <- stri_replace_all_fixed(bad_line, '+', '')
bad_line <- stri_replace_all_fixed(bad_line, '/', '')
bad_line <- stri_replace_all_fixed(bad_line, '-', '')
bad_line <- stri_replace_all_fixed(bad_line, ';', '')
bad_line <- stri_replace_all_fixed(bad_line, '!', '')
bad_line <- stri_replace_all_fixed(bad_line, ',', '')
bad_line <- stri_split_fixed(tolower(bad_line), ' ')[[1]]
bad_line <- sort(unique(bad_line))
dput(bad_line)

ngrams <- c()
for(i in seq_along(bad_line)){
  for(j in i:length(bad_line)){
    ngrams <- c(ngrams, paste(bad_line[i], bad_line[j]))
  }
}

cf[,`Feature Name` := stri_replace_all_fixed(`Feature Name`, 'NGRAM_OCCUR_L2_desc-', '')]
cf[,exact := as.integer(`Feature Name` %in% c(bad_line, ngrams))]

cf[,`Feature Name` := stri_replace_all_fixed(`Feature Name`, 'NGRAM_OCCUR_L2_desc-', '')]
cf[,sim := sapply(`Feature Name`, function(x) max(stringsim(x, bad_line, method='lcs')))]
cf <- cf[order(-exact, -sim),]

cf[exact==1,list(`Feature Name`, Coefficient)]


cf[exact==1,sum(Coefficient)]

# iconv -c -f utf-8 -t ascii zach/lc363.csv > zach/lc363_clean.csv
# iconv -c -f utf-8 -t ascii zach/lc900.csv > zach/lc900_clean.csv
