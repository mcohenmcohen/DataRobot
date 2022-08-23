library(readr)
library(data.table)
library(stringi)
#https://stackoverflow.com/questions/32567743/is-there-anyway-to-read-dat-file-from-movielens-to-r-studio

# Read movies
movies <- data.table(read_delim('~/workspace/data-science-scripts/zach/ml-1m/movies.dat', delim='::', col_names=F))[,c(1,3,5),with=F]
setnames(movies, c("movie_id", "title", "genre"))
movies[,genre := stri_split_fixed(genre, '|')]
for(i in 1:5){
  var = sapply(movies[['genre']], '[', i)
  var[is.na(var)] <- 'NA'
  set(movies, j=paste0('genre', i), value=var)
}
movies[,genre := NULL]
movies[,movie_id := paste0('m', movie_id)]

# Read Genres
users <- data.table(read_delim('~/workspace/data-science-scripts/zach/ml-1m/users.dat', delim='::', col_names=F))[,c(1,3,5,7,9),with=F]
setnames(users, c("user_id", "gender", "age", "occupation", "zip"))
users[,user_id := paste0('u', user_id)]
users[,age := paste0('a', age)]
users[,occupation := paste0('o', occupation)]
users[,zip := paste0('z', zip)]

# Read Ratings
# ratings <- data.table(read_delim('~/workspace/data-science-scripts/zach/ml-1m/ratings.dat', delim='::', col_names=F))
ratings <- fread("~/workspace/data-science-scripts/zach/ml-1m/ratings.dat", sep=":", select=c(1,3,5,7))
setnames(ratings, c("user_id", "movie_id", "rating", "timestamp"))

# Save dataset from autoint paper
# https://arxiv.org/pdf/1810.11921.pdf
# MovieLens-1M This dataset contains users
# ratings on movies. During binarization, we treat samples with a
# rating less than 3 as negative samples because a low score indicates
# that the user does not like the movie. We treat samples with a rating
# greater than 3 as positive samples and remove neutral samples, i.e.,
# a rating equal to 3.
ratings_bin = copy(ratings)
ratings[,timestamp := NULL]
ratings[,user_id := paste0('u', user_id)]
ratings[,movie_id := paste0('m', movie_id)]
ratings[,rating := sign(rating-3)]
ratings <- ratings[rating != 0,]
ratings[,rating := (rating+1)/2]
ratings[,table(rating)]
fwrite(ratings, '~/Downloads/movielens_1M_binary.csv')

# Add more features
dat <- ratings
dat <- merge(dat, users, by='user_id', all.x=T, all.y=F)
dat <- merge(dat, movies, by='movie_id', all.x=T, all.y=F)
fwrite(dat, '~/Downloads/movielens_enriched_1M_binary.csv')