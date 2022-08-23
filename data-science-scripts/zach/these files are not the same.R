a <- fread('~/Downloads/v2_pred_df.csv')
b <- fread('~/Downloads/batch_pred_df.csv')

a <- a[c(16906, 31514, 31354),]
b <- b[c(16906, 31514, 31354),]

setdiff(names(a), names(b))
setdiff(names(b), names(a))

x <- sapply(a, function(x) sum(is.na(x))); sum(x)
x <- sapply(b, function(x) sum(is.na(x))); sum(x)

x <- sapply(a, function(x) sum(is.nan(x))); sum(x)
x <- sapply(b, function(x) sum(is.nan(x))); sum(x)

for(n in names(a)){
  comp = a[[n]]==b[[n]]

  check_1 = all(comp[is.finite(comp)])
  check_2 = all(which(!is.finite(a[[n]])) == which(!is.finite(b[[n]])))

  check = check_1 & check_2

  if(!check){
    print(n)
    print(summary(a[[n]]))
    print(summary(b[[n]]))
    #stop('One of these is not like the other.')

    #idx = head(which(comp==FALSE), 1)
    #print(idx)
    print(sprintf("%.60f", a[[n]]))
    print(sprintf("%.60f", b[[n]]))
  }
}

