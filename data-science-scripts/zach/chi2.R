ivans_test <- function(data){
  frequency_j_i = data['cluster 1', 'has "ngram"'] / data['cluster 1', 'total documents']
  frequency_j_not_i = (data['other clusters', 'has "ngram"']+10) / (data['other clusters', 'total documents'] * 3)
  return(frequency_j_i / frequency_j_not_i)
}

book_stats = cbind(c(50, 0), c(1000, 1000))
row.names(book_stats) <- c('cluster 1', 'other clusters')
colnames(book_stats) <- c('has "ngram"', 'total documents')
fisher.test(book_stats, conf.level = .95)[['conf.int']][1]
chisq.test(book_stats)[['statistic']]

car_stats = cbind(c(1000, 50), c(10000, 10000))
row.names(car_stats) <- c('cluster 1', 'other clusters')
colnames(car_stats) <- c('has "ngram"', 'total documents')
ivans_test(car_stats)
fisher.test(car_stats)[['conf.int']][1]
chisq.test(car_stats)[['statistic']]

book_stats = cbind(c(50, 0, 0), c(10000, 5000, 5000))
row.names(book_stats) <- c('cluster 1', 'cluster 2', 'cluster 3')
colnames(book_stats) <- c('has "ngram"', 'total documents')
chisq.test(book_stats)[['statistic']]

car_stats = cbind(c(1000, 25, 25), c(10000, 5000, 5000))
row.names(car_stats) <- c('cluster 1', 'cluster 2', 'cluster 3')
colnames(car_stats) <- c('has "ngram"', 'total documents')
chisq.test(car_stats)[['statistic']]
