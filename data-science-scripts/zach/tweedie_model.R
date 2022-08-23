# Load Libraries and set seed
library(statmod)
library(tweedie)
library(data.table)
set.seed(42)

# Choose a tweedie p
# Try 1.5 vs 1.0
TWEEDIE_P <- 1.5

# Load the data and take a random 20% holdout
dat <- fread('~/workspace/data-science-scripts/zach/tweedie_example.csv')
dat <- dat[order(runif(.N)),]

split <- dat[,as.integer(.N*.80)]
dat_train <- dat[1:split,]
dat_test  <- dat[(split + 1):.N,]

# Fit a tweedie glm, using exposure as an exposure
tweedie_glm_exposure <- dat_train[,glm(
  ClaimAmount ~ Idade_Pessoa_Segura_Anuidade + Num_Participantes,
  family=tweedie(var.power=TWEEDIE_P, link.power=0),
  offset=log(Exposure),
  data=.SD
)]
round(coef(summary(tweedie_glm_exposure)), 2)

# Fit a tweedie glm, using exposure as a weight
tweedie_glm_weight <- dat_train[,glm(
  ClaimAmount / Exposure ~ Idade_Pessoa_Segura_Anuidade + Num_Participantes,
  family=tweedie(var.power=TWEEDIE_P, link.power=0),
  weight=Exposure,
  data=.SD
)]
round(coef(summary(tweedie_glm_weight)), 2)

# Predict from both models
pred_exposure <- predict(tweedie_glm_exposure, newdata=dat_test, exposure=dat_test[,Exposure], type='response')
pred_weight <- predict(tweedie_glm_weight, newdata=dat_test, type='response') * dat_test[,Exposure]

# Compare the means of the 2 models
# Note that the "Exposure" model is very biased
mean(pred_exposure)  # 86.36801
mean(pred_weight)  # 36.24433
dat_test[,mean(ClaimAmount)]  # 33.90641

# Compare the tweedie deviance of the 2 models
# Note that the "exposure" model has lower tweedie p
dat_test[,mean(tweedie.dev(ClaimAmount, pred_exposure, TWEEDIE_P))]  # 61.44561
dat_test[,mean(tweedie.dev(ClaimAmount, pred_weight, TWEEDIE_P))]  # 67.55139

# Compare the RMSE of the 2 models - RMSE MEASURES BIAS
# Note that the "exposure" model has lower tweedie p
rmse <- function(a, b) sqrt(mean((a - b) ^ 2))
dat_test[,rmse(ClaimAmount, pred_exposure)]  # 489.8187
dat_test[,rmse(ClaimAmount, pred_weight)]  # 476.699

# Compare the 2 model's coefficients
# Note the coefficients are VERY similar
# **Note the intercept from the exposure model is higher:***
# tweedie p = 1.5 biases the model towards higher predictions
coef(tweedie_glm_exposure) - coef(tweedie_glm_weight)

# Conclusion 1: using exposure as an exposure leads to less bias
# Conclusion 2: using exposure as an exposure leads to lower tweedie loss
# Conclusion 3: changing TWEEDIE_P to 1.0 causes both methods to yield identical results
