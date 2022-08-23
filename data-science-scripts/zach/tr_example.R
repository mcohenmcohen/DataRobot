library(jsonlite)
x = list(
  'Matrix of word-grams occurrences tokenizer'= 'scikit-learn based tokenizer',
  'Matrix of word-grams occurrences binary'= TRUE,
  'Matrix of word-grams occurrences sublinear_tf'= FALSE,
  'Matrix of word-grams occurrences use_idf'= FALSE,
  'Matrix of word-grams occurrences norm'= 'L2',
  'Total weights'= 447,
  'Intercept'= -1.20494260954,
  'Model precision'= 'single',
  'Loss distribution'= 'Binomial Deviance',
  'Link function'= 'logit',
  'Pairwise interactions found'=list(
    list(var1='groups', var2='item_ids', cf=0.0087981938365420381),
    list(var1='item_ids', var2='こんにちは', cf=0.0073953000639295765),
    list(var1='tvh', var2='こんにちは', cf=0.0062485787413982278),
    list(var1='dates_2 (Year)', var2='こんにちは', cf=0.0039294478075596277),
    list(var1='dates (Year)', var2='groups', cf=0.0035171332815932552),
    list(var1='dates_2 (Year)', var2='groups', cf=0.0028919212094156216),
    list(var1='dates (Year', var2='tvh', cf=0.0027262013972543125),
    list(var1='dates (Year)', var2='item_ids', cf=0.0087981938365420381),
    list(var1='dates (Year)', var2='こんにちは', cf=0.0018645497034906681),
    list(var1='dates_2 (Year)', var2='tvh', cf=0.0013746505632680347)
  )
)
toJSON(x, pretty=T, auto_unbox=T)
