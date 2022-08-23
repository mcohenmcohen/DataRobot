#Load Data
library('shrinkR')
library('reshape2')
library('ggplot2')
library('data.table')

# 55cde23dc8e2f45a84fc45c0 - 3 grid search, autp for reg only
# 55ce2de699154b692d05ea74 - no grid search, auto for class and reg
# 55ce3bb2bb805e0d61dfd64d - 10 grid search, auto for class and reg, no regular rulefile
# 55d39f57c8e2f42c125b9b44 - xavier grid search, auto reg vs rulefit reg (no class)
# 55d3e73ac8e2f402f63bea22 - Same as prev, but bigger machines
# 55d4e41548a5d27d7af4bc6e - More iters, constant learning
# 55d50d3f87508459dbf9fe47 - Same as prev, with param tweaks
# 55d620f0c8e2f454fe0c24cd - No main effects, bigger datasets #FAILED
# 55d629ed99154b3ae67764e1 - 100 batch size #FAILED
# 55d629f737d67c6b885a1c79 - 1,000 batch size #FAILED
# 55d629f7aef3125fe520468e - 10,000 batch size #FAILED
# 55d629f815eea36df26ab5a2 - 100,000 batch size #FAILED
# 55e4ba0e3a822e0d7d58117e - No main effects, bigger datasets #FINALL WORKED

#check <- verifyYaml('~/workspace/datarobot/tests/ModelingMachine/out_of_core_rulefit_datasets_largedata_regonly.yaml')

hash <- '55e4ba0e3a822e0d7d58117e'
lb <- loadLeaderboard(hash)
setnames(lb, make.names(names(lb)))
lb[, alpha := sapply(Best_Parameters_P1, function(x){
  if(is.null(x$alpha)) NA
  else x$alpha
})]
lb[,table(Y_Type, alpha)]
lb[,length(unique(Filename))]
lb[,length(unique(Filename)), by='Y_Type']

#Subset data
dat <- copy(lb)
dat <- dat[,list(
  Sample_Size,
  model_name,
  Y_Type,
  Filename,
  Project_Metric,
  Total_Time_P1,
  Gini.Norm_P1,
  Max_RAM_GB,
  RMSE_H,
  holdout_scoring_time,
  error_P1)]
dat[,model_name := gsub('[C|R] $', '', model_name)]

#Shorten filenames
dat[,Filename := gsub('.csv', '', Filename, fixed=TRUE)]
for(x in list(
  c("All_Players_All_Years_wconf_b_t_ht_wt_actpos_scouting_predict_isML", 'All_Players_All_Years_isML'),
  c("All_Players_All_Years_wconf_b_t_ht_wt_pos_predict_role", 'All_Players_All_Years_role'),
  c("forrest-wine-lasso-selected-vars-additional-columns2-wine-userID", 'forrest-wine-lasso'),
  c("FIXED_3_year_stats_for_Juniors_150p_PA_predict_role", 'FIXED_3_year_stats_predict_role'),
  c("250p_PA_HS_3_years_since_debut_predict_70p", '250p_PA_HS_3_years_predict_70p'),
  c("cemst-decision-prediction2-asr3_train", 'cemst-asr3'),
  c('cemst-decision-prediction2-asr2_train', 'cemst-asr2'),
  c('cemst-decision-prediction-asr0_train', 'cemst-asr0'),
  c('correct_s1__eg2008jk_train_converted', 'correct_s1_eg2008jk'),
  c('crys_1vsrest_88atoms_train_converted', 'crys_1vsrest_88atoms'),
  c('cburford-phrase_unique_match_train', 'cburford-phrase'),
  c('blakhol_rg_test_5_train_converted', 'blakhol_rg_test_5'),
  c('28_Features_split_train_converted', '28_Features_split'),
  c('AirlineFlights2008-reduced-70000', 'AirlineFlights2008-11k'),
  c('AirlineFlights2008-reduced-35000', 'AirlineFlights2008-5k')
)){
  set(dat, i=which(dat$Filename == x[1]), j='Filename', value=x[2])
}
tail(dat[,unique(Filename)[order(nchar(unique(Filename)))]])

#Add rows
dat[,Filename := paste0(Filename, ' (1e', floor(log10(Sample_Size)), ')')]

#Hack factor levels
factor_levels <- unique(dat[,list(Filename, Y_Type, Sample_Size)])
factor_levels <- factor_levels[order(Y_Type, Sample_Size), Filename]
dat[,Filename := factor(Filename, levels=factor_levels)]
dat[,Y_Type := factor(Y_Type)]

#Quick and dirty comparison function
compare <- function(x, what=c('Gini.Norm_P1', 'Max_RAM_GB', 'Total_Time_P1', 'holdout_scoring_time')){
  x <- copy(x)
  x <- melt(x[,c('Sample_Size', 'model_name', 'Y_Type', 'Filename', what), with=FALSE], measure.vars=what)
  x <- dcast.data.table(x, Filename + Sample_Size + Y_Type + variable ~ model_name)
  ggplot(x, aes(
    x=OOCRULEFIT, y=RULEFIT,
    size=log10(Sample_Size),
    )) +
    geom_point(alpha = 0.33) +
    geom_abline(slope=1,linetype="dashed") +
    facet_wrap(variable~Y_Type, ncol=2, scales='free') +
    theme_bw() + coord_fixed() +
    theme(legend.position = "bottom")
}

#Make plots
outfile <- paste0('~/Desktop/ooc_rulefit_compare-',hash,'-.pdf')
pdf(outfile, width=8.5, height=11)
compare(dat)

#Binary
# for(var in c('Total_Time_P1', 'Max_RAM_GB', 'Gini.Norm_P1')){
#   plt <- simpleGraph(
#     dat[Y_Type == 'Binary',],
#     x=var, y='Filename',
#     shape = 'Y_Type',
#     col = 'model_name',
#     title = paste('Binary -', var),
#     facet_x='', facet_y='') +
#     theme(axis.text.y = element_text(size=6))
#   print(plt)
# }

#Regression
for(var in c('Total_Time_P1', 'Max_RAM_GB', 'Gini.Norm_P1')){
  plt <- simpleGraph(
    dat[Y_Type == 'Regression',],
    x=var, y='Filename',
    shape = 'Y_Type',
    col = 'model_name',
    title = paste('Regression -', var),
    facet_x='', facet_y='') +
    theme(axis.text.y = element_text(size=6))
  print(plt)
}

#Close plot and inspect
sink <- dev.off()
system(paste('open', outfile))
