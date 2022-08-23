library(readr)
library(data.table)

# Download https://s3.amazonaws.com/datarobot_public_datasets/stack_overflow_closed_question_1Gb.csv
dat_raw <- read_csv('~/datasets/stack_overflow_closed_question_1Gb.csv')
dat <- data.table(dat_raw)
dat[,PostClosedDate := NULL]
write_csv(dat, '~/datasets/stack_overflow_closed_question_1Gb.csv')
