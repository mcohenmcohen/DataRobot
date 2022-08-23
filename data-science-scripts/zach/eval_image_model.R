library(data.table)
x <- fread('Downloads/man_vs_woman_2_Elastic-Net_Classifier_(L1___Binomial_Deviance)_(123)_100_Informative_Features (1).csv')
setnames(x, 'Cross-Validation Prediction', 'pred')
t <- fread('~/Downloads/filename_target_map_woman_male_247.csv')
x[,pred2 := round(pred,2)]

x[,min := as.integer(pred==min(pred)), by='target']
x[,max := as.integer(pred==max(pred)), by='target']

x[(max & target == 'woman') | (min & target == 'male'),]
x[(min & target == 'woman') | (max & target == 'male'),]
setorderv(x, 'pred')
x


library(data.table)
x <- fread('Downloads/catdog_Nystroem_Kernel_SVM_Classifier_(36)_100_Informative_Features (1).csv')
setnames(x, 'Cross-Validation Prediction', 'pred')

x[,min := as.integer(pred==min(pred)), by='target']
x[,max := as.integer(pred==max(pred)), by='target']

x[(max & target == 'dog') | (min & target == 'cat'),]
x[(min & target == 'dog') | (max & target == 'cat'),]

setorderv(x, 'pred')
x

pdir <- '~/datasets/Buses/positives/'
ndir <- '~/datasets/Buses/negatives/'
odir <- '~/datasets/Buses/named/'

negatives <- list.files(ndir)
positives <- list.files(pdir)

negatives <- negatives[grepl('.jpg', negatives, fixed=T)]
positives <- positives[grepl('.jpg', positives, fixed=T)]

for(f in negatives){
  file.copy(paste0(ndir, f), paste0(odir, 'not-', f), overwrite=T)
}
for(f in positives){
  file.copy(paste0(pdir, f), paste0(odir, 'bus-', f), overwrite=T)
}


library(data.table)
x <- fread('Downloads/Bus_Support_Vector_Classifier_(Linear_Kernel)_(87)_79.9_f1.csv')
setnames(x, 'Cross-Validation Prediction', 'pred')

x[,min := as.integer(pred==min(pred)), by='target']
x[,max := as.integer(pred==max(pred)), by='target']

x[(max & target == 'notbus') | (min & target == 'zbus'),]
x[(min & target == 'notbus') | (max & target == 'zbus'),]

setorderv(x, 'pred')
x
