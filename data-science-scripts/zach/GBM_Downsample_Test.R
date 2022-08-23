#Load Data
library('shrinkR')
library('ggplot2')
library('data.table')
lb <- loadLeaderboard('55c24958993a8947f7e565e4')
setnames(lb, make.names(names(lb)))

#Open plot
outfile <- '~/Desktop/gbm_downsample_results.pdf'
pdf(outfile, width=11, height=8.5)

#Just time
simpleGraph(
  lb[Y_Type == 'Binary',], 'model_name', 'Total_Time_P1', title='Binary models total time',
  facet_x = 'Filename',
  facet_y = 'Project_Metric') +
  theme(axis.text.x = element_text(angle = 45)) +
  expand_limits(y=0)
simpleGraph(
  lb[Y_Type == 'Regression'], 'model_name', 'Total_Time_P1', title='Regression models total time',
  facet_x = 'Filename',
  facet_y = 'Project_Metric') +
  theme(axis.text.x = element_text(angle = 45)) +
  expand_limits(y=0)

#Gini norm
simpleGraph(
  lb[Y_Type == 'Binary',], 'model_name', 'Gini.Norm_P1', title='Binary models gini norm',
  facet_x = 'Filename',
  facet_y = 'Project_Metric') +
  theme(axis.text.x = element_text(angle = 45)) +
  expand_limits(y=0)
simpleGraph(
  lb[Y_Type == 'Regression'], 'model_name', 'Gini.Norm_P1', title='Regression models gini norm',
  facet_x = 'Filename',
  facet_y = 'Project_Metric') +
  theme(axis.text.x = element_text(angle = 45)) +
  expand_limits(y=0)

#Gini norm
simpleGraph(
  lb[Y_Type == 'Binary',], 'model_name', 'Max_RAM_GB', title='Binary models max RAM (Gb)',
  facet_x = 'Filename',
  facet_y = 'Project_Metric') +
  theme(axis.text.x = element_text(angle = 45)) +
  expand_limits(y=0)
simpleGraph(
  lb[Y_Type == 'Regression'], 'model_name', 'Max_RAM_GB', title='Regression models max RAM (Gb)',
  facet_x = 'Filename',
  facet_y = 'Project_Metric') +
  theme(axis.text.x = element_text(angle = 45)) +
  expand_limits(y=0)

#REMOVE BAD MODELS
#lb <- lb[!Filename %in% c('grockit_train_full.csv', 'chargeback.csv'),]

#Just error
simpleGraph(
  lb[Y_Type == 'Binary',], 'model_name', 'error_P1', title='Binary models error P1',
  facet_x = 'Filename',
  facet_y = 'Project_Metric') +
  theme(axis.text.x = element_text(angle = 45)) +
  expand_limits(y=0)
simpleGraph(
  lb[Y_Type == 'Regression'], 'model_name', 'error_P1', title='Regression models error P1',
  facet_x = 'Filename',
  facet_y = 'Project_Metric') +
  theme(axis.text.x = element_text(angle = 45)) +
  expand_limits(y=0)

#Time vs error
simpleGraph(
  lb[Y_Type == 'Binary',], 'Total_Time_P1', 'error_P1', title='Binary models error vs runtime',
  facet_x = 'Filename',
  facet_y = 'Project_Metric')
simpleGraph(
  lb[Y_Type == 'Regression'], 'Total_Time_P1', 'error_P1', title='Regression models error vs runtime',
  facet_x = 'Filename',
  facet_y = 'Project_Metric')

#Close plot
sink <- dev.off()
system(paste('open', outfile))
