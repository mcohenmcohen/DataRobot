training_data <- -100:-10
training_min <- min(training_data)
training_range <- max(training_data) - min(training_data)
training_data_scaled <- (training_data - training_min) / training_range
summary(training_data_scaled)

new_data <- -11:1
new_data_scaled <- (new_data - training_min) / training_range
new_data_scaled
new_data_scaled

new_data_scaled_and_capped <- pmax(pmin(new_data_scaled, 1), 0)
new_data_scaled_and_capped
