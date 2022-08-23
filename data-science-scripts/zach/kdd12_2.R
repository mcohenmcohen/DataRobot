stop()
rm(list=ls(all=T))
gc(reset=T)

library(data.table)
library(fastmatch)
library(stringi)

# Load data
kdd12 <- fread("~/Downloads/track2/training.txt", sep="\t")
setnames(kdd12, c('click', 'impression', 'display_url', 'adid', 'advertiser_id', 'depth', 'position', 'query_id', 'keyword_id', 'title_id', 'description_id', 'user_id'))

# Load user data
userid_profile <- fread("~/Downloads/track2/userid_profile.txt", sep="\t")
setnames(userid_profile, c('user_id', 'gender', 'age'))

# Join the user data
setkeyv(kdd12, 'user_id')
setkeyv(userid_profile, 'user_id')
kdd12 = userid_profile[kdd12,]

# Binarize clicks
kdd12[kdd12$click > 0, click := 1]

# Convert numerics to categorical
categorize = function(x){
  gc(reset=T)
  table = sort(unique(x))
  map = fmatch(x, table)
  return(stri_paste('x', map))
}
for(v in c('display_url', 'adid', 'advertiser_id', 'depth', 'position', 'query_id', 'keyword_id', 'title_id', 'description_id', 'user_id', 'gender', 'age')){
  print(v)
  set(kdd12, j=v, value=categorize(kdd12[[v]]))
}

# Randomly order the data
set.seed(42)
kdd12[,order := runif(.N)]
setorderv(kdd12, 'order')
kdd12[,order := NULL]
gc(reset=T)

# Write out a little less than 5GB of data
TARGET_SIZE <- 4.295e+9
NROWS <- 62924002
size_str <- tolower(gsub(' ', '_', utils:::format.object_size(TARGET_SIZE, units='Gb'), fixed=T))
outfile <- paste0('~/Downloads/kdd_', size_str, '.csv')
fwrite(head(kdd12, NROWS), outfile)

# Use the size of the outfile to compute the rows we should ACTUALLY use
fsize <- file.size(outfile)
utils:::format.object_size(fsize, units='Gb')
bytes_per_row = fsize/NROWS
target_rows = TARGET_SIZE/bytes_per_row
print(target_rows)
