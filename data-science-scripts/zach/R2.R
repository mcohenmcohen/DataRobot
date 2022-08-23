calc_r2 <- function(rmse, mean_model_rmse, mean_model_r2){

  baseline_rmse = sqrt(mean_model_rmse^2 / (1 - mean_model_r2))
  r2 = 1 - rmse^2 / baseline_rmse^2

  return(r2)
}

calc_r2(rmse=2.6021e+4, mean_model_rmse=3856.8448, mean_model_r2=-2.31e-3)  # -44.6245
calc_r2(rmse=5.6361e+12, mean_model_rmse=8.3172e+12, mean_model_r2=-12.3352)  # -5.1236
