library(data.table)
dat = fread('~/Downloads/modeling_machine_predictive_testing.csv')
for (l in dat[,unique(pr_link)]){
  browseURL(l)
}