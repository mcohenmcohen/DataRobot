examine_file <- function(x){
  dat <- fread(paste("curl --silent", x, "| head -n 11"))
  return(
    list(
      x=x,
      dim=dim(dat),
      str=ls.str(dat)
    )
  )
}

for(u in c(
  "https://s3.amazonaws.com/dataintake/10GBFiles/LPH07_2_dirty.csv",
  "https://s3.amazonaws.com/dataintake/10GBFiles/SLC14_dirty.csv",
  "https://s3.amazonaws.com/dataintake/10GBFiles/LPH07_dirty.csv",
  "https://s3.amazonaws.com/dataintake/10GBFiles/SLC14_2_dirty.csv",
  "https://s3.amazonaws.com/dataintake/10GBFiles/criteo_10gb.csv",
  "https://s3.amazonaws.com/dataintake/10GBFiles/airlines_10gb.csv",
  "https://s3.amazonaws.com/dataintake/10GBFiles/gsod_1929_2009_10GB.csv"
)){
  print(examine_file(u))
}
