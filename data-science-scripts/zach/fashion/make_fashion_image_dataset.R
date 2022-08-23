# Setup
stop()
rm(list=ls(all=T))
gc(reset = T)
library(data.table)
library(pbapply)
library(stringi)
library(kit)
set.seed(227503)

# Notes:
# efficientnet-b0-pruned

# Load data files
img_cat <- fread('list_category_img.txt')
cat_map <- fread('list_category_cloth.txt', fill=T, skip=1)

# Remove some categories
remove <- c(
  'Halter',
  'Coverup',
  'Kimono',
  'Nightdress',
  'Onesie',
  'Top',
  'Tank',
  'Sarong',
  'Shorts',
  'Sweatshorts',
  'Trunks'
)
cat_map[,category_label := 1:.N]
keep <- cat_map[!(category_name %in% remove), category_label]
cat_map <- cat_map[category_label %in% keep,]
img_cat <- img_cat[category_label %in% keep,]

# Take a random sample of images
img_cat <- img_cat[sample(.N, 1e5),]

# Determine image folders
folder_list <- img_cat[, stri_split_fixed(image_name, '/')]
img_cat[,folder := sapply(folder_list, '[', 2)]
img_cat[,image := sapply(folder_list, '[', 3)]

# Map categories mean likes
# cat_map[,dput(category_name)]
manual_likes <- list(
  "Anorak"=3,
  "Blazer"=3,
  "Blouse"=4,
  "Bomber"=2,
  "Button-Down"=2,
  "Cardigan"=8,
  "Flannel"=10,
  "Henley"=8,
  "Hoodie"=0.001,
  "Jacket"=5,
  "Jersey"=.1,
  "Parka"=7,
  "Peacoat"=12,
  "Poncho"=15,
  "Sweater"=3,
  "Tee"=.01,
  "Turtleneck"=3,
  "Capris"=3,
  "Chinos"=4,
  "Culottes"=4,
  "Cutoffs"=.01,
  "Gauchos"=10,
  "Jeans"=3,
  "Jeggings"=0.001,
  "Jodhpurs"=8,
  "Joggers"=10,
  "Leggings"=4,
  "Skirt"=6,
  "Sweatpants"=0.001,
  "Caftan"=11,
  "Cape"=15,
  "Coat"=4,
  "Dress"=5,
  "Jumpsuit"=5,
  "Kaftan"=9,
  "Robe"=2,
  "Romper"=5,
  "Shirtdress"=1,
  "Sundress"=15)
for(category in names(manual_likes)){
  i = cat_map[,which(category_name==category)]
  set(cat_map, i=i, j='mean_likes_cat', value=manual_likes[[category]])
}
cat_map[,max(mean_likes_cat)]

# Merge the means onto the main data and sample from the poisson for counts
key <- 'category_label'
setkeyv(img_cat, key)
setkeyv(cat_map, key)
img_cat <- merge.data.table(img_cat, cat_map, by='category_label')
img_cat[,likes := rpois(.N, mean_likes_cat)]
summary(img_cat)
img_cat[,round(table(likes) /.N * 100, 1)]

# Delete and re-create outdir
outdir <- 'output/'
unlink(outdir, recursive=TRUE)
dir.create(outdir)

# Make output folders
unique_likes <- img_cat[,sort(funique(likes))]
for(like in unique_likes){
  dir.create(paste0(outdir, like))
}

# Move the images to the correct output folder
setkeyv(img_cat, 'likes')
success_rate <- pbsapply(unique_likes, function(l){
  old_file_list <- img_cat[likes == l, image_name]
  new_file_list <- img_cat[likes == l, stri_paste(outdir, l, '/', folder, '_', image)]
  sink <- file.copy(old_file_list, new_file_list)
  sum(sink) / length(sink)
})
table(round(success_rate*100, 1))

# Zip the images
outzip <- 'fashion.zip'
unlink(outzip)
zip(zipfile = outzip, files = outdir)
