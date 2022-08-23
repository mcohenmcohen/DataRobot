library(data.table)
set.seed(42)
ratingTableFileLocation = '~/Downloads/Untitled_Project_Generalized_Additive2_Model_(65)_64.04_Informative_Features_rating_table.csv'
ratingTableHeader = readLines(ratingTableFileLocation)
index = which(grepl('^Feature Name,Feature Strength,', ratingTableHeader))
ratingTableHeader <- ratingTableHeader[1:(index-1)]
ratingTableFile = fread(ratingTableFileLocation, skip=index-1)
ratingTableFile
N = nrow(ratingTableFile)
R = runif(N)
set(ratingTableFile, j='Coefficient\r', value = paste0(R, '\r'))
modifiedFile = '~/Downloads/mod2.csv'
cat(ratingTableHeader, file=modifiedFile, sep='\r\n')
fwrite(ratingTableFile, modifiedFile, append=T, col.names=T)
system('head -n50 ~/Downloads/mod2.csv')
