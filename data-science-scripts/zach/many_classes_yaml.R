
# Setup
stop("STOP!")
rm(list=ls(all=T))
gc(reset=T)
library(data.table)
library(yaml)

# Known 45+ classes
classes <- fread('~/workspace/data-science-scripts/zach/multiclas_yaml_class_count.csv')
classes <- classes[classes>=45,][order(classes),]
keep <- classes[['dataset_name']]

# More known 45 classes
keep <- c(
  keep,
  'https://s3.amazonaws.com/datarobot_public_datasets/openML/datasets/spectrometer_v1.csv',
  'https://s3.amazonaws.com/datarobot_public_datasets/openML/datasets/amazon-commerce-reviews_v1.csv',
  'https://s3.amazonaws.com/datarobot_public_datasets/openML/datasets/one-hundred-plants-margin_v1.csv',
  'https://s3.amazonaws.com/datarobot_public_datasets/openML/datasets/one-hundred-plants-shape_v1.csv',
  'https://s3.amazonaws.com/datarobot_public_datasets/openML/datasets/one-hundred-plants-texture_v1.csv',
  'https://s3.amazonaws.com/datarobot_public_datasets/openML/datasets/BachChoralHarmony_v1.csv'
)
keep <- unique(keep)

# Make yaml
yaml <- yaml.load_file('~/workspace/mbtest-datasets/mbtest_datasets/data/Functionality/mbtest_data_multiclass.yaml')

# Subset
keep_idx <- which(sapply(yaml, '[[', 'dataset_name') %in% keep)
yaml <- yaml[keep_idx]

# New files
new_names <- expand.grid(
  c('reddit_top', 'aloi'),
  c('10.csv', '100.csv', '1000.csv')
)
new_names <- paste(new_names[[1]], new_names[[2]], sep='_')

new_yaml <- lapply(new_names, function(x){
  list(
    dataset_name = paste0('https://s3.amazonaws.com/datarobot_public_datasets/', x),
    target = ifelse(grepl('subreddit', x), 'subreddit', 'target'),
    multiclass = TRUE
  )
})

# Save
out <- c(yaml, new_yaml)
OUTFILE <- "~/workspace/mbtest-datasets/mbtest_datasets/data/Functionality/multiclass_45_to_100.yaml"
cat(as.yaml(out), file=OUTFILE)
system(paste('head -n50', OUTFILE))
