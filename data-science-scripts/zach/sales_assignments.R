# Solution based on the "warehouse" problem solution here:
# https://dirkschumacher.github.io/ompr/articles/problem-warehouse-location.html
# 
# This problem is very likely a well-nown integer programming problem, and there's probably a solution we could find somewhere
# But whatever, let's build it ourselves

rm(list=ls(all=T))
gc(reset=T)

library(data.table)
library(ompr)
library(magrittr)
library(ompr.roi)
library(ROI.plugin.glpk)

dat <- fread('~/workspace/data-science-scripts/zach/play_along.csv')
dat[,distance_km_sqr := distance_km**2] # Hypothesis: travel pain increases quadratically with distance

dat[,compoany_key_int := as.integer(factor(compoany_key))]
dat[,employee_int := as.integer(factor(employee))]

N_employees = dat[,max(employee_int)]
N_companies = dat[,max(compoany_key_int)]

setkeyv(dat, c('employee_int', 'compoany_key_int'))

cost <- function(employee_id, company_id) {
  dat[list(employee_id, company_id), distance_km_sqr]
}

print(cost(1, 3))

model <- MIPModel() %>%
  # 1 iff employee i gets assigned to company j
  add_variable(x[i, j], i = 1:N_employees, j = 1:N_companies, type = "binary") %>%
  
  # maximize the preferences
  set_objective(sum_expr(cost(i, j) * x[i, j], i = 1:N_employees, j = 1:N_companies), "min") %>%
  
  # every company needs to be assigned to an employee
  add_constraint(sum_expr(x[i, j], j = 1:N_companies) == 1, i = 1:N_employees)

model

result <- solve_model(model, with_ROI(solver = "glpk", verbose = TRUE))
sum(result$solution)
a=data.table(get_solution(result, x[i,j]))