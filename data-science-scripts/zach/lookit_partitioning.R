library(data.table)
a <- fread('~/Downloads/LC_-_clone_after_models_-_PARENT_Gradient_Boosted_Trees_Classifier_with_Early_Stopp_(97)_100_Informative_Features.csv')
b <- fread('~/Downloads/LC_-_clone_after_models_-_CHILD_Gradient_Boosted_Trees_Classifier_with_Early_Stopp_(97)_100_Informative_Features.csv')

all(a[['row_id']] == b[['row_id']])
all(a[['Partition']] == b[['Partition']])
