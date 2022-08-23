MAE = function(a, b) mean(abs(a-b))

# Target
y = c(1, 10, 100)

# Model 1:
MAE(y, mean(y))

# Model 2:
MAE(y, exp(mean(log(y))))

# Incorrec model 2
exp(MAE(log(y), mean(log(y))))
