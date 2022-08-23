library(data.table)
grade_pred=fread('~/Downloads/grade.csv')[[1]]
sub_grade_pred=fread('~/Downloads/sub_grade.csv')[[1]]

cor(grade_pred, sub_grade_pred)
cor(qlogis(grade_pred), qlogis(sub_grade_pred))
cor(plogis(grade_pred), plogis(sub_grade_pred))
