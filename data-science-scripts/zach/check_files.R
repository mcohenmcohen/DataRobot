library(data.table)

a = fread('~/datasets/DR_Demo_LendingClub_Guardrails_BEST_01.csv')
b = fread('~/datasets/DR_Demo_LendingClub_Guardrails_BEST_yesno.csv')

setdiff(names(a), names(b))
setdiff(names(b), names(a))
dim(a) - dim(b)

name_diff = function(x){
  out = a[[x]] == b[[x]]
  out_na = is.na(a[[x]]) == is.na(b[[x]])
  out[is.na(out)] = out_na[is.na(out)]
  return(sum(!out))
}
x = sapply(names(a), name_diff)
x[x!=0]
