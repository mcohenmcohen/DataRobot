# Setup
rm(list=ls(all=T))
dev.off()
gc(reset=T)
set.seed(42)
library(compiler)
library(data.table)
library(Matrix)

# Old logic for n_components
old_logic <- cmpfun(function(n_samples, n_features){
  smallest_dim <- pmin(n_samples, n_features)

  max_dim <- round(0.5 * smallest_dim)
  n_components <- round(n_features ** 0.5)
  n_components <- pmin(n_components, max_dim)

  return(n_components)
})
stopifnot(old_logic(c(4, 100, 100), c(200, 2, 100)) == c(2, 1, 10))

# New logic for n_components
new_logic <- cmpfun(function(n_samples, n_features){
  smallest_dim <- pmin(n_samples, n_features)
  n_components <- rep(NA, length(smallest_dim))

  test_1 <- smallest_dim <= 4
  n_components[test_1] <- smallest_dim[test_1]

  test_2 <- (!test_1) & (n_samples <= 10 | n_features <= 25)
  n_components[test_2] <- 5

  test_3 <- (!test_1) & (!test_2)
  n_components[test_3] <- round(pmin(n_samples[test_3] * 0.5, n_features[test_3] ** 0.5))

  return(n_components)
})
stopifnot(new_logic(c(4, 100, 100), c(200, 2, 100)) == c(4, 2, 10))

# Test
test_cases <- CJ(n_samples=1:1000, n_features=1:1000)
test_cases[,min_dim := pmin(n_samples, n_features)]
test_cases[,old_k := old_logic(n_samples, n_features)]
test_cases[,new_k := new_logic(n_samples, n_features)]
test_cases[,diff_k := new_k-old_k]
test_cases[,pct_diff_k := diff_k/old_k]
test_cases[diff_k!=0, stopifnot(all(n_samples < 10 | n_features <= 20))]
test_cases[n_samples==1000 & diff_k != 0,]
test_cases[diff_k == max(diff_k),]
summary(test_cases[diff_k == max(diff_k),])
summary(test_cases[n_samples > 100 & n_features>100,])
test_cases[,list(pct=round(.N/nrow(test_cases)*100, 1)), by='diff_k'][order(diff_k),]

# Plot
img_mat <- test_cases[diff_k!=0,drop0(sparseMatrix(i=n_samples, j=n_features, x=diff_k/max(diff_k)))]
image(img_mat)

# Iterate through some real datasets:
mbtest_ids <- c(
  '5f98c7c48047520acaabbf1c', # 7.3 dockerized large
  '5f98c2fb23325218fd819873', # 7.3 dockerized small
  '61c507e1a6a3f9ed27375960'  # Current with preds
)
mbtest_urls <- paste0(
  "http://shrink.drdev.io/api/leaderboard_export/advanced_export.csv?mbtests=",
  mbtest_ids,
  "&max_sample_size_only=false"
)
dat_list <- lapply(mbtest_urls, fread)

dat <- rbindlist(dat_list, fill=T, use.names = T)
#sort(names(dat))
dat <- dat[,list(
  n_samples=as.numeric(Sample_Size),
  num=as.numeric(dataset_x_numeric),
  cat=as.numeric(dataset_x_cat),
  txt=as.numeric(dataset_x_txt),
  img=as.numeric(dataset_x_image)
)]
dat <- dat[is.finite(n_samples),]
dat <- dat[is.finite(num) | is.finite(cat) | is.finite(txt) | is.finite(img),]
dat <- unique(dat)
dat[,n_features := num +
      cat * 100 +
      txt * 200000 +
      img * 1024]
dat[,old_n_components := old_logic(n_samples, n_features)]
dat[,new_n_components := new_logic(n_samples, n_features)]
dat[,diff_n_components := new_n_components - old_n_components]
dat[,pct_diff_n_components := new_n_components / diff_n_components]
dat[,diff_data_size := n_samples * new_n_components - n_samples * old_n_components]
dat[,pct_diff_data_size := n_samples * new_n_components / diff_data_size]

# Plots
plot_dat <- unique(dat[,list(new_n_components, old_n_components)])
plot_dat[,plot(log(new_n_components) ~ log(old_n_components))]
abline(0, 1)
plot_dat[,plot(new_n_components ~ old_n_components)]
abline(0, 1)
plot_dat[old_n_components < 10, plot(new_n_components ~ old_n_components)]
abline(0, 1)

# Stats
summary(dat)
dat[diff_n_components!=0,]
summary(dat[diff_n_components!=0,])
summary(dat[diff_n_components!=0 & diff_data_size > 10000,])
dat[diff_n_components!=0 & diff_data_size > 10000,][order(diff_data_size),]
