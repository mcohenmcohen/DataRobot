
# Setup
library(data.table)
TARGET_SIZE <- 1.074e+9
size_str <- tolower(gsub(' ', '_', utils:::format.object_size(TARGET_SIZE, units='Gb'), fixed=T))
print(size_str)
N <- 92918152

# Load data
data <- fread('https://s3.amazonaws.com/datarobot_public_datasets/wild_function.csv', nrows=N)
utils:::format.object_size(object.size(data), units='Gb')

# Save data
outfile <- paste0('~/wild_function_',size_str ,'.csv')
print(outfile)
fwrite(data, outfile)
fsize <- file.size(outfile)
utils:::format.object_size(fsize, units='Gb')

# Calculate size needed
bytes_per_row = fsize/N
target_rows = TARGET_SIZE/bytes_per_row
print(target_rows)
