#Clean Data
library(data.table)
library(readr)
library(caret)
dat = fread('~/datasets/dr_training.csv', header=FALSE)
setnames(dat, c('Desc', 'Price', 'Category', 'Target'))
dat[,Target := Target + 1]
dat[, table(Target)]
write_csv(dat, '~/datasets/dr_training_clean.csv')
dat[, table(Category, Target)]

dat[Target == 1,][1:5,]
dat[Target == 2,][1:5,]
dat[Target == 3,][1:5,]

#Eval predicitons
VW <- fread('~/datasets/Multiclass_test_Vowpal_Wabbit_Multiclass_Regressor_(One_Against_Al_(45)_80_Informative_Features_holdout.csv')
GBM <- fread('~/datasets/Multiclass_test_Multinomial_Gradient_Boosted_Trees_Regressor_(46)_80_Informative_Features_holdout.csv')
Blend <- fread('Multiclass_test_AVG_Blender_(449+445)_(454)_80_Informative_Features_holdout.csv')

eval_cm <- function(x){
  x <- x[Partition == 'Holdout',]
  print(confusionMatrix(x[[2]]-1, round(x[[4]]-1)))
}

eval_cm(VW)
eval_cm(GBM)
eval_cm(Blend)

rows <- VW[round(`Cross-Validation Prediction`) != Target,row_id]
dat[rows+1,]

#Reason codes
rc <- fread('~/datasets/Multiclass_test_Vowpal_Wabbit_Multiclass_Regressor_(One_Against_Al_(45)_80_Informative_Features_RC_3_gt_1.000.csv')
setorderv(rc, 'row_id')
dat[rows+1,]
rc[rows+1,]
