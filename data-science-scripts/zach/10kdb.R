library(datarobot)
library(data.table)
dat = fread('~/datasets/10kDiabetes.csv')
dat[,readmitted := as.numeric(readmitted)]
dat[,table(readmitted)]
project <- SetupProject(dataSource=dat)
up <- UpdateProject(project, workerCount = 20, holdoutUnlocked = TRUE)
st <- SetTarget(project = project, target = "readmitted", positiveClass=0)
ViewWebProject(project)
fwrite(dat, '~/datasets/10kDiabetes_num_target.csv')
