library(data.table)
dat <- fread('~/workspace/data-science-scripts/zach/Binary_data_pipelines.csv')

sum(!is.na(dat)) / prod(dim(dat))

pl <- dat[,Pipeline]
dat[,Pipeline := NULL]

mat <- as.matrix(dat)
row.names(mat) <- pl


mat <- scale(mat, center = T, scale = T)
working_mat <- copy(mat)
idx <- is.na(mat)
working_mat[idx] <- 0

for(i in 1:1){
  model <- prcomp(working_mat, retx=T)
  pred <- as.vector(model$x[!idx])
  act <- as.vector(mat[!idx])
  err <- sqrt(mean((pred - act)^2))
  print(err)
  
  working_mat <- copy(mat)
  working_mat[idx] <- model$x[idx]
}

summary(as.vector(mat))
x=mat[1:10,1:10]; row.names(x) <- NULL; colnames(x) <- NULL; print(x)

x=working_mat; x[idx]=NA
round(summary(as.vector(x)), 2)
x=round(x[1:10,1:10], 2); row.names(x) <- NULL; colnames(x) <- NULL; print(x)

out <- data.table(
  dataset=colnames(mat)
)
for(i in 1:nrow(out)){
  out[,best_model := row.names(mat)[which.max(mat[,i])]]
}

#TODO: RESHAPE TALL AND RUN A BAYES GLM WITH MAIN EFFECTS AND INTERACTIONS