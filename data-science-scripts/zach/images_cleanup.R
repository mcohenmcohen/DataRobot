stop()
rm(list=ls(all=T))
gc(reset=T)
library(data.table)
library(stringi)
library(base64enc)

##################################################################
# Functions
##################################################################

# https://stackoverflow.com/questions/46032969/how-to-display-base64-images-in-r
# https://stackoverflow.com/questions/27828842/r-add-title-to-images-with-rasterimage
# https://stackoverflow.com/questions/14660372/common-main-title-of-a-figure-panel-compiled-with-parmfrow
to_image <- function(x){
  plot.new()
  raw <- base64decode(x)
  img <- jpeg::readJPEG(raw)
  plot(as.raster(img))
  rasterImage(img, 0, 0, 1, 1, interpolate=FALSE)
}

image_plot <- function(x){
  par(mfrow=c(5, 5), mar=c(0, 0, 0, 0), xaxs = 'i', yaxs='i')
  for(i in 1:length(x)){
    to_image(x[i])
  }
}

make_collage <- function(x, targetname, imagename, targetlevel, title=TRUE){
  set.seed(42)
  idx <- which(x[[targetname]] == targetlevel)
  idx <- sample(idx, 12)
  image_plot(x[idx,][[imagename]])
  if(title){
    title(targetlevel, line=-3, outer=TRUE, cex.main=5)
  }
}

##################################################################
# Rentals
##################################################################

# https://s3.amazonaws.com/datarobot_public_datasets/images/train_text_image.csv
x <- fread('~/datasets/train_text_image.csv')

setnames(x, paste0('image', 1:9), paste0('url', 1:9))
setnames(x, 'images', 'urls')

x[,c('V1', 'id', 'listing_id', 'interest_level', 'display_address', 'urls') := NULL]
x[,paste0('url', 1:9) := NULL]

x <- x[price > 250 & price < 10000,]
fwrite(x, '~/datasets/rental_prices.csv')

x_disp <- copy(x)
x_disp[,description := stri_sub(description, 1, 50)]

x_disp[which.min(price),list(price, street_address, description, bedrooms, bathrooms, latitude, longitude, features)]
x_disp[which.min(price),to_image(image0)]

x_disp[1,list(price, street_address, description, bedrooms, bathrooms, latitude, longitude, features)]
x_disp[1,to_image(image0)]

x_disp[14290,list(price, street_address, description, bedrooms, bathrooms, latitude, longitude, features)]
x_disp[14290,to_image(image0)]

x_disp[which.max(price),list(price, street_address, description, bedrooms, bathrooms, latitude, longitude, features)]
x_disp[which.max(price),to_image(image0)]

##################################################################
# Butterflies
##################################################################

fly <- fread('https://s3.amazonaws.com/datarobot_public_datasets/images/train_butterflies_image.csv')
dev.off()
for(class in fly[,sort(unique(class))]){
  print(class)
  make_collage(fly, 'class', 'data', class)
  cat("Press Enter to continue...")
  invisible(readline(prompt="Press [enter] to continue"))
}
fly[,table(class)]
##################################################################
# Drivers
##################################################################

a <- fread('https://s3.amazonaws.com/datarobot_public_datasets/images/driver_imgs_data.csv', header=T)
for(class in a[,sort(unique(classname))]){
  print(class)
  make_collage(a, 'classname', 'img', class)
  cat("Press Enter to continue...")
  invisible(readline(prompt="Press [enter] to continue"))
}
a[,table(classname)]
a[,table(subject)]

##################################################################
# Simpsons
##################################################################

b <- fread('https://s3.amazonaws.com/datarobot_public_datasets/images/train_simpsons_image.csv', header=T)
for(class in b[,rev(names(tail(sort(table(class)), 7)))]){
  print(class)
  make_collage(b, 'class', 'data', class, title=TRUE)
  cat("Press Enter to continue...")
  invisible(readline(prompt="Press [enter] to continue"))
}

b[,sort(unique(class))]

##################################################################
# Chest xrays
##################################################################

b <- fread('https://s3.amazonaws.com/datarobot_public_datasets/images/train_chest_xray.csv', header=T)
for(class in b[,sort(unique(class))]){
  print(class)
  make_collage(b, 'class', 'data', class, title=TRUE)
  cat("Press Enter to continue...")
  invisible(readline(prompt="Press [enter] to continue"))
}

b[,sort(unique(class))]

