# Setup
rm(list=ls(all=T))
gc(reset = T)
library(data.table)
library(pbapply)
library(stringi)

# Make output dir
outdir <- '/Users/zachary/Downloads/retail_sales/'
if(file.exists(outdir)){
  unlink(outdir, recursive=TRUE)
}
dir.create(outdir)

# Load and subset data
# https://www.kaggle.com/jmmvutu/summer-products-and-sales-in-ecommerce-wish
dat <- fread('/Users/zachary/Downloads/archive (1)/summer-products-with-rating-and-performance_2020-08.csv')
dat <- dat[retail_price > 10,]
dat[,table(units_sold)]

# Download images
downloads <- pblapply(1:nrow(dat), function(i){
  sales <- dat[i, units_sold]
  sales_dir <- paste0(outdir, sales, '/')
  if(!file.exists(sales_dir)){
    dir.create(sales_dir)
  }
  file_url <- dat[i, product_picture]
  file_name <- rev(stri_split_fixed(file_url, '/')[[1]])[1]
  download.file(file_url, paste0(sales_dir, file_name), mode = 'wb')
})
