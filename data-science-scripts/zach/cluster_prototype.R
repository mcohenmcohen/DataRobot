library(fpc)
library(data.table)
library(ggplot2)
library(grid)
library(gridExtra)

#For developing
# raw_dat <- fread('https://s3.amazonaws.com/datarobot_data_science/test_data/10k_diabetes_train80.csv')
# pca_dim=50
# tsne_dim=2
# verbose=TRUE

colors_1 <- "#1f78b4"
colors_2 <- c(
  "#ff7f00", "#6a3d9a", "#33a02c", "#e31a1c", "#b15928",
  "#a6cee3", "#fdbf6f", "#cab2d6", "#b2df8a", "#fb9a99", "black",
  "grey"
)
colors_2 <- rep(colors_2, 100)
colors <- c(colors_1, colors_2)

#Run clustering
tsne_cluster <- function(raw_dat, pca_dim=50, tsne_dim=2, verbose=TRUE){
  library(data.table)
  library(irlba)
  library(stringi)
  library(Rtsne)
  library(fpc)
  library(kknn)

  if(verbose) message('Setting up')
  db <- data.table(raw_dat)
  length_unique <- sapply(db, function(x) length(unique(x)))
  keep <- (length_unique > 1) & (length_unique < .99 * nrow(db))
  db <- db[,keep,with=FALSE]
  db_matrix <- sparse.model.matrix(~ 0 + ., db)

  if(verbose) message('Starting PCA')
  pca <- irlba(
    db_matrix, nu=min(ncol(db)-2, nrow(db)-2, pca_dim), nv=0,
    verbose=verbose, center=colMeans(db_matrix))$u

  if(verbose) message('Starting TSNE')
  tsne <- Rtsne::Rtsne(pca, tsne_dim, pca=FALSE, check_duplicates=FALSE, verbose=verbose)

  if(verbose) message('Starting initial DBscan embedding')
  bad_clustering <- TRUE

  backoff <- .95
  increase <- 1.05
  min_pts <- max(nrow(db) * 0.001, 25)
  eps <- 1
  max_iters <- 100
  iters <- 0
  while(bad_clustering & (iters < max_iters)){
    dbs_cluster <- dbscan::dbscan(tsne$Y, eps, minPts=min_pts, splitRule='suggest', borderPoints=TRUE)
    cluster_table <- table(dbs_cluster$cluster)
    if(cluster_table['0'] / length(dbs_cluster$cluster) > .10){
      if(verbose) message('Too many noise points.  Increasing')
      eps <- eps * increase
    } else if(max(cluster_table) / length(dbs_cluster$cluster) > .25){
      if(verbose) message('Too many points in one cluster.  Backing Off')
      eps <- eps * backoff
    } else if(sum(cluster_table < .05 * nrow(db)) / length(cluster_table) > .50){
      if(verbose) message('Too many small clusters. Increasing')
      min_pts <- min_pts + 1
      eps <- eps * increase
    } else {
      if(verbose) message('Clustering successful!')
      bad_clustering <- FALSE
    }
    iters <- iters + 1
  }
  if(bad_clustering & verbose) message('Clustering Failed =(')

  if(verbose) message('Assigning outliers to centers')
  clusters <- data.table(id=1:nrow(tsne$Y), tsne$Y, cluster=dbs_cluster$cluster, key='cluster')
  outliers <- clusters[cluster==0,]
  non_outliers <- clusters[cluster!=0,]

  clusters[,outlier := as.integer(cluster == 0)]
  fit.kknn <- kknn(cluster ~ V1 + V2, train=non_outliers, test=outliers, k=1, kernel='rectangular')
  clusters[cluster==0, cluster := as.integer(fit.kknn$fit)]

  if(verbose) message('Calculating outlier scores')
  clusters[outlier == 1,cluster := as.integer(fit.kknn$fit)]
  clusters[,dist := as.numeric(NA)]
  clusters[outlier == 1, dist := fit.kknn$D]

  if(verbose) message('Assembling output')
  setorderv(clusters, 'id')
  top_outliers <- head(clusters[order(dist, decreasing=TRUE),], 10)
  top_outliers[,outlier := NULL]
  top_outliers_dt <- db[top_outliers$id,]
  clusters[,cluster := factor(cluster)]
  list(
    clusters = clusters,
    top_outliers = data.table(top_outliers, top_outliers_dt),
    data_with_clusters = data.table(
      cluster=dbs_cluster$cluster,
      db
    )
  )
}

#TSNE Cluster Plot
cluster_plot <- function(tsne_res, title){
  set.seed(42)

  library(ggplot2)
  library(scales)

  #Points
  point_dat <- data.table(
    x = tsne_res$clusters$V1,
    y = tsne_res$clusters$V2,
    cluster = tsne_res$clusters$cluster,
    outlier = tsne_res$clusters$outlier
  )
  point_plot <- ggplot(point_dat, aes(x=x, y=y, color=cluster)) +
    geom_point(alpha=0.25, aes(shape=factor(outlier)))

  #Convext hulls
  clusters <- sort(unique(point_dat$cluster))
  hulls <- lapply(clusters, function(x){
    i <- which(point_dat[['cluster']] == x)
    point_dat_subset <- point_dat[i,]
    point_dat_subset <- point_dat_subset[outlier == 0,]
    i_hull <- point_dat_subset[,chull(x, y)]
    point_dat_subset[i_hull,]
  })
  hulls <- rbindlist(hulls)
  hull_plot <- point_plot + geom_polygon(data = hulls, alpha=0, size=1.5)

  #Centers
  center_dat <- point_dat[outlier == 0,list(
    x = mean(x),
    y = mean(y),
    .N
  ), by='cluster']
  center_plot <- hull_plot + geom_label(
    data=center_dat,
    aes(x=x, y=y, color=cluster, label=cluster),
    size=5
  )

  #Final formatting
  final_plot <- center_plot + theme_bw() + xlab('TSNE1') + ylab('TSNE2') +
    scale_color_manual(values=colors) + ggtitle(title) +
    guides(colour=FALSE, size=FALSE, shape=FALSE)
  final_plot
}

#Outlier table
#http://stackoverflow.com/a/35384795/345660
table_plot <- function(x, title=''){
  library(grid)
  library(gridExtra)
  library(gtable)

  tbl <- tableGrob(x, rows=NULL, theme=ttheme_default(base_size = 8, padding = unit(c(2, 2), "mm")))

  title <- textGrob(title, gp=gpar(fontsize=14))
  padding <- unit(5,"mm")

  tbl <- gtable_add_rows(
    tbl,
    heights = grobHeight(title) + padding,
    pos = 0)
  tbl <- gtable_add_grob(tbl, title, 1, 1, 1, ncol(tbl))

}
outlier_table_plot <- function(tsne_res){
  library(grid)
  library(gridExtra)
  library(gtable)

  pd <- copy(tsne_res$top_outliers)
  pd[,V1 := NULL]
  pd[,V2 := NULL]
  first <-  c('row', 'out_score', 'clust')
  setnames(pd, c('id', 'dist', 'cluster'), first)
  pd_count_diff <- sort(sapply(pd, function(x) length(unique(x))), decreasing=TRUE)
  data.table::setcolorder(pd, names(pd_count_diff))
  last <- setdiff(names(pd), first)
  pd <- pd[,c(first, head(last, 10)), with=FALSE]
  pd <- pd[,lapply(.SD, function(x){
    if(is.numeric(x)){
      x <- round(x, 2)
      x <- signif(x, digits = 2)
    }
    return(x)
  })]
  pd[,row := NULL]
  tbl <- table_plot(pd, "Top Outliers")

  return(tbl)
}

#Means vs overall means for one cluster
fast_mode <- function(x){
  library(data.table)
  out <- data.table(x, key='x')
  out <- out[,list(.N), by='x']
  setorder(out, -N, x)
  return(out[1,x])
}
len_unique <- function(x) length(unique(x))

cluster_summary_plot <- function(tsne_res, c=2){

  #Variable importance
  library(ranger)
  db <- copy(tsne_res$data_with_clusters)
  db[,cluster := as.integer(cluster == c)]
  rf <- ranger(cluster ~ ., db, importance='permutation')
  imp <- sort(importance(rf), decreasing=TRUE)
  imp <- imp[imp>0]
  imp <- imp / max(imp)
  plot_dat <- data.table(
    var = factor(names(imp), levels=names(imp)),
    imp = imp
  )
  plot_dat <- data.table(
    var = factor(names(imp), levels=rev(names(imp))),
    imp = imp
  )
  var_imp_plot <- ggplot(plot_dat, aes(x=var, y=imp)) +
    geom_bar(stat="identity") +
    xlab('') + theme_bw() + ggtitle(paste('Cluster', c, 'Importance')) +
    theme(axis.text.y = element_text(angle = 15, hjust = 1)) + coord_flip()

  #Variable summary
  db_sum <- db[,lapply(.SD, function(x){
    if(is.numeric(x)){
      stats::median(x)
    }
    fast_mode(x)
  }), by='cluster']
  db_sum <- db_sum[,c('cluster', names(imp)),with=FALSE]
  keep <- sapply(db_sum, function(x) length(unique(x))) > 1
  db_sum <- db_sum[,keep,with=F]
  db_sum[,cluster := factor(
    cluster, levels=0:1, labels=c(
      'All Others', paste('Cluster', c)
    ))]
  if(ncol(db_sum) >  6){
    db_sum <- db_sum[,1:6,with=FALSE]
  }

  #Return double plot
  arrangeGrob(
    nrows=2, heights = c(9, 3),
    grobs = list(
      ggplotGrob(var_imp_plot),
      table_plot(db_sum, paste('Cluster', c, 'Summary'))
    )
  )
}

#Combine all into one plot
run_it_all <- function(raw_dat, name){

  #Remove constant
  unique <- sapply(raw_dat, function(x) length(unique(x)))
  raw_dat <- raw_dat[, unique > 1, with=FALSE]

  #NA impite and char to factor
  for(var in names(raw_dat)){
    x <- raw_dat[[var]]
    if(is.numeric(x)){
      x[is.na(x)] <- median(x, na.rm=TRUE)
    } else{
      x <- factor(x)
      x <- addNA(x, ifany = TRUE)
    }
    set(raw_dat, j=var, value=x)
  }

  tsne_dat <- tsne_cluster(raw_dat)
  tsne_plot <- cluster_plot(tsne_dat, name)

  outlier_table <- outlier_table_plot(tsne_dat)

  cluster_to_plot <- tsne_dat$top_outliers[,list(d=sum(dist)), by='cluster'][which.max(d), cluster]
  right_half <- cluster_summary_plot(tsne_dat, c=cluster_to_plot)

  left_half <- arrangeGrob(
    grobs = list(
      ggplotGrob(tsne_plot),
      outlier_table),
    nrows=2, heights = c(9, 4))

  print({
    grid.arrange(
      grobs = list(left_half, right_half),
      ncols=2,
      widths=c(10, 5)
    )
  })
}

pdf('~/Documents/clustering_mocks.pdf', width=20, height=15)

#10k DB
db_10k <- fread('https://s3.amazonaws.com/datarobot_data_science/test_data/10k_diabetes_train80.csv')
db_10k[,c('diag_1_desc', 'diag_2_desc', 'diag_3_desc') := NULL]
db_10k[,c('discharge_disposition_id') := NULL]
#db_10k[,c('diag_1', 'diag_2', 'diag_3') := NULL]
run_it_all(db_10k, '10k Diabetes')

#Lending club
LC <- fread('https://s3.amazonaws.com/datarobot_data_science/test_data/10K_2007_to_2011_Lending_Club_Loans_v2_mod.csv')
LC[, desc := NULL]
LC[, title := NULL]
LC[, earliest_cr_line := NULL]
LC[, emp_title := NULL]
run_it_all(LC, 'Lending Club')

#Boston housing
BH <- fread('https://s3.amazonaws.com/datarobot_data_science/test_data/boston_housing.csv')
run_it_all(BH, 'Boston Housing')

#Diamonds
DM <- fread('https://s3.amazonaws.com/datarobot_data_science/test_data/diamonds.csv')
run_it_all(DM, 'Diamonds')

#cars
KC <- fread('https://s3.amazonaws.com/datarobot_data_science/test_data/kickcars_train_full_test20.csv')
KC[,RefId := NULL]
KC[,PurchDate := NULL]
setnames(KC, gsub('MMRAcquisition', '', names(KC)))
setnames(KC, gsub('MMRAcquisiton', '', names(KC)))
setnames(KC, gsub('MMRCurrent', '', names(KC)))
KC <- KC[,!duplicated(names(KC)),with=FALSE]
run_it_all(KC, 'Kick Cars')

#Movies
MV <- fread('https://s3.amazonaws.com/datarobot_data_science/test_data/movies.csv')
MV[,V1 := NULL]
setnames(MV, make.names(names(MV)))
run_it_all(MV[year>2000,], 'Movies')

#Open pdf
dev.off()
system('open ~/Documents/clustering_mocks.pdf')
