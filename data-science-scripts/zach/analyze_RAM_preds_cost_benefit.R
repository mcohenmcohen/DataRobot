# Reset
rm(list=ls(all=T))
gc(reset=T)

# Libraries
library(data.table)

# Load Data
data <- fread('~/workspace/data-science-scripts/zach/RAM_test_RAM_Time_Preds.csv')
data <- data[!is.na(Max_clock_time),]

# Analysis Inputs
cost_per_gb_per_hour <- 4.256/480  # Assume 8 60GB containers on an r4.16xlarge
prediction_coefficient <- 3
prediction_constant <- 0
container_sizes <- c(
  61440,
  30720,
  20480,
  10240,
  5120)
container_sizes <- sort(container_sizes)

# Calculate RAM used under new logic
data[,buffered_ram := Prediction * prediction_coefficient + prediction_constant]
data[,new_container_id := findInterval(buffered_ram, c(0, container_sizes), all.inside = T)]
data[,table(new_container_id)]
data[,new_ram := container_sizes[new_container_id]]

# Calculate Old Cost
# Don't we use 30GB containers a lot of the time though?
data[,old_cost := 61440/1000 * Max_clock_time/3600 * cost_per_gb_per_hour]

# Calculate new cost
data[,new_cost := new_ram/1000 * Max_clock_time/3600 * cost_per_gb_per_hour]

# If we OOM, assume we pay the new cost while OOMing
# And then pay the old cost after we OOM and jump to 60GM
data[buffered_ram < Target_Max_Ram, new_cost := new_cost + old_cost]

# Summarize savings
total_savings <- data[,sum(old_cost) - sum(new_cost)]
pct_savings <- total_savings / data[,sum(old_cost)]
print(total_savings)
print(pct_savings)

# Angriest customer
# An extra day of fit time seems bad
data[buffered_ram < Target_Max_Ram,summary(Max_clock_time)]
data[buffered_ram < Target_Max_Ram,max(Max_clock_time) / (60*60*24)]  
