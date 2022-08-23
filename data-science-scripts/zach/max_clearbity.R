x <- fread('~/Downloads/Clearbit-fraud-DR-prospects-training-set2.csv')
hist(x[['Tier Level']])
mean(x[['Tier Level']])

a <- fread('~/Downloads/Clearbit_Project_AVG_Blender_(32+33+31)_(71)_64.02_Informative_Features_Clearbit-fraud-DR-prospects-test-set2 (1).csv')
b <- fread('~/Downloads/Clearbit_Project_AVG_Blender_(32+33+31)_(71)_64.02_Informative_Features (1).csv')

setorderv(b, 'RowId')
b[,Actual := x[['Tier Level']]]

setnames(b, make.names(names(b)))
b[Partition == '0.0', Partition := 'Validation']

b_agg <- b[,list(rmse = sqrt(mean((Actual - Cross.Validation.Prediction)^2))), by='Partition']
setorderv(b_agg, 'Partition')
b_agg


write.csv(b, '~/datasets/max_clearbit_PvA.csv', row.names=FALSE)
