library(yaml)

V2 <- yaml.load_file('~/workspace/DataRobot/ModelingMachine/engine/tasks2/task_desc_and_url.yaml')
V1 <- yaml.load_file('~/workspace/DataRobot/ModelingMachine/engine/tasks/task_desc_and_url.yaml')

V1_not_in_V2 <- setdiff(names(V1), names(V2))

V1_new <- c(V2, V1[V1_not_in_V2])
V1_new <- as.yaml(V1_new)
cat(V1_new, file='~/workspace/DataRobot/ModelingMachine/engine/tasks/task_desc_and_url.yaml')
