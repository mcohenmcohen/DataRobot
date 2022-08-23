data(iris)
class <- paste0(
  ifelse(iris$Species == 'versicolor', 1, -1),
  ' |f',
  ' sl:', iris$Sepal.Length,
  ' sw:', iris$Sepal.Width,
  ' pl:', iris$Petal.Length,
  ' pw:', iris$Petal.Width
)
reg <- paste0(
  iris$Sepal.Length,
  ' |f',
  ' ', iris$Species,
  ' sw:', iris$Sepal.Width,
  ' pl:', iris$Petal.Length,
  ' pw:', iris$Petal.Width
)
class <- paste(class, collapse='\n')
reg <- paste(reg, collapse='\n')

#cat(class)
#cat(reg)

classfile <- path.expand('~/datasets/testvw.txt')
regfile <- path.expand('~/datasets/vw_reg.txt')
cat(class, file=classfile)
cat(reg, file=regfile)

#Class
base_call <- paste('vw -d', classfile, '--passes 1', '--loss_function hinge --final_regressor vw_model.vw')
system(base_call)
system(paste(base_call, '--nn 10 --meanfield'))
system(paste(base_call, '--boosting 1'))
system(paste(base_call, '--boosting 10'))
system(paste(base_call, '--ksvm --kernel linear --l2 1'))
system(paste(base_call, '--ksvm --kernel poly --degree 2 --l2 1'))
system(paste(base_call, '--ksvm --kernel poly --degree 3 --l2 1'))
system(paste(base_call, '--autolink 2'))
system(paste(base_call, '--autolink 3'))
system(
  paste(
    paste('vw -d', classfile, '--final_regressor vw_model.vw --loss hinge'),
    '--kernel linear --random_seed 1234 --ksvm --learning_rate 0.001 --decay_learning_rate 0.99 --quantile_tau 0.5 --power_t 0.5 --l2 10 --loss_function logistic --bit_precision 18 --passes 100 --cache_file vw_cache
    ')
  )

system(paste(base_call, '--boosting 10 --alg adaptive --gamma 100'))
system(paste('vw -i vw_model.vw --link logistic -r vwpreds.txt -t -d', classfile))
system('cat vwpreds.txt')


#Reg
base_call <- paste('vw -d', regfile, '--passes 1', '--loss_function squared --final_regressor vw_model.vw')
system(base_call)
system(paste(base_call, '--nn 10'))
system(paste(base_call, '--boosting 1'))
system(paste(base_call, '--boosting 10'))
system(paste(base_call, '--autolink 2'))
system(paste(base_call, '--autolink 3'))



system(paste('vw -i vw_model.vw -p vwpreds.txt -t -d', regfile))
system('cat vwpreds.txt')
