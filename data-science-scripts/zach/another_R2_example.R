library(data.table)

load_and_clean = function(x){
  data = fread(paste0('~/Downloads/', x))
  setnames(data, tolower(make.names(names(data))))
  setkey(data, partition)
  return(data)
}

a = load_and_clean('new_customer_PC2_first_60_days_61dda32217c78c6ac58_Gradient_Boosted_Trees_Regressor_(Least-Squares_Lo_(38)_38.4_Informative_Features_validation_and_holdout.csv')

b = load_and_clean('new_customer_PC2_first_60_days_61dda32217c78c6ac58_Gradient_Boosted_Trees_Regressor_(Least-Squares_Lo_(38)_38.4_Informative_Features.csv')

rmse = function(actual, predicted) sqrt(mean((actual - predicted) ** 2))
sse = function(actual, predicted) sum((actual - predicted) ** 2)
R2 = function(actual, predicted) 1 - sse(actual, predicted) / sse(actual, mean(actual))

a[,rmse(amt_pc_2, cross.validation.prediction), by='partition']
a[,R2(amt_pc_2, cross.validation.prediction), by='partition']

b[,rmse(amt_pc_2, cross.validation.prediction), by='partition']
b[,R2(amt_pc_2, cross.validation.prediction), by='partition']

keys = c('row_id', 'partition')
setkeyv(a, keys)
setkeyv(b, keys)

x = merge.data.table(a, b, by=keys, all=T)
x[partition=='0.0' & is.na(cross.validation.prediction.x),]
x[partition=='0.0' & is.na(cross.validation.prediction.y),]
