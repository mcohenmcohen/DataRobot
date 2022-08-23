library(Deriv)

# Linear reg
Deriv(~sum((y - (m %*% x + b)) ^ 2), c('m', 'b'))

# Hinge reg
relu = function(x) ifelse(x>0, x, 0)
Deriv(~sum((y - m*relu(x + b)) ^ 2), c('m', 'b'))

# Linear Hinge reg
relu = function(x) ifelse(x>0, x, 0)
Deriv(~sum((y - (a %*% x + b + c %*% relu(x - d))) ^ 2), c('a', 'b', 'c', 'd'))

# Linear Hinge reg - simpler variant
relu = function(x) ifelse(x>0, x, 0)
Deriv(~sum((y - (a %*% x + b + c %*% relu(x - d))) ^ 2), c('a', 'b', 'c', 'd'))
