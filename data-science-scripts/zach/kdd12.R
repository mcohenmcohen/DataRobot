library(data.table)
library(fastmatch)
library(stringi)

# Load data
kdd12 <- fread("~/Downloads/track2/training.txt", sep="\t")
userid_profile <- fread("~/Downloads/track2/userid_profile.txt", sep="\t")
setnames(kdd12, c('click', 'impression', 'display_url', 'adid', 'advertiser_id', 'depth', 'position', 'query_id', 'keyword_id', 'title_id', 'description_id', 'user_id'))
setnames(userid_profile, c('user_id', 'gender', 'age'))

# Convert numerics to categorical
categorize = function(x){
  gc(reset=T)
  table = sort(unique(x))
  map = fmatch(x, table)
  return(stri_paste('x', map))
}
kdd12[,display_url := categorize(display_url)]
kdd12[,adid := categorize(adid)]
kdd12[,advertiser_id := categorize(advertiser_id)]
kdd12[,depth := categorize(depth)]
kdd12[,position := categorize(position)]
kdd12[,query_id := categorize(query_id)]
kdd12[,keyword_id := categorize(keyword_id)]
kdd12[,title_id := categorize(title_id)]
kdd12[,description_id := categorize(description_id)]
kdd12[,user_id := categorize(user_id)]

# Write out ~5GB of the raw data
set.seed(42)
kdd12[,order := runif(.N)]
setorderv(kdd12, 'order')
kdd12[,order := NULL]
gc(reset=T)

TARGET_SIZE <- 5e+9
size_str <- tolower(gsub(' ', '_', utils:::format.object_size(TARGET_SIZE, units='Gb'), fixed=T))
print(size_str)
N <- 78998840
outfile <- paste0('~/Downloads/kdd_',size_str ,'.csv')
print(outfile)
fwrite(head(kdd12, N), outfile)
fsize <- file.size(outfile)
utils:::format.object_size(fsize, units='Gb')
bytes_per_row = fsize/N
target_rows = TARGET_SIZE/bytes_per_row
print(target_rows)

# Join the user data
setkeyv(kdd12, 'user_id')
setkeyv(userid_profile, 'user_id')
kdd12_joined = userid_profile[kdd12,]
