



# Loading packages --------------------------------------------------------

library(datarobot)
library(reshape2)
library(mclust)

# Connecting to DataRobot -------------------------------------------------

setwd("/Users/taylor.larkin/Documents/DRU R Class")
ConnectToDataRobot(configPath = "./drconfig.yaml")

# Grab project via link ---------------------------------------------------

project.object <- GetProject("5c65468bfd2b983641b6477c")

# Grabbing data -----------------------------------------------------------

wine.data <- read.csv("./wine_data.csv", sep=";")
colnames(wine.data) <- gsub("\\.", "_", colnames(wine.data))
wine.data$wine_is_good <- as.factor(ifelse(wine.data$quality >= 7, "good", "bad"))

# Reproducing testing
set.seed(10)
testing <- wine.data[-createDataPartition(wine.data$wine_is_good, p = 0.9, list = FALSE),]

# Grab feature impact and prediction explanations -------------------------

# Best model
best.model <- GetRecommendedModel(project.object, type = RecommendedModelType$MostAccurate)

# Best model's feature impact
featureImpact <- GetFeatureImpactForModel(best.model)

scoring <- ListPredictionDatasets(project.object)[[1]]
pe.request <- RequestReasonCodes(best.model, scoring$id, maxCodes = 10)
pe.request.metadata <- GetReasonCodesMetadataFromJobId(project.object, pe.request)
pe.frame <- GetAllReasonCodesRowsAsDataFrame(project.object, pe.request.metadata$id)

# Clustering prediction explanations --------------------------------------

# Creating a dataset of feature strength scores
out <- list()
for(i in 1:length(grep("FeatureName", colnames(pe.frame)))){
  tmp <- pe.frame[,c(paste0("reason", i, "FeatureName"),
                     paste0("reason", i, "Strength"))]
  colnames(tmp) <- c("name", "strength")
  out[[i]] <- cbind.data.frame("rowId" = pe.frame$rowId, tmp)
}
strength.matrix <- dcast(melt(out, id = c('rowId', "name")), rowId+variable ~ name)[,-c(1,2)]
strength.matrix[is.na(strength.matrix)] <- 0

# Using K-means clustering in prediction explanations
# Choosing 3 for simplicity here
clustering.algo.pe <- kmeans(strength.matrix, centers = 3)

# Plot with scatterplot of 2 most useful features (for viz purposes only)
# Note only 3 clusters and 2 features are picked, so the mixture of colors is to be expected
pairs(testing[,featureImpact$featureName[1:2]], col=clustering.algo.pe$cluster,
      main = "Raw Data with Clusters")

# Comparing back to clustering on raw data --------------------------------

# Standardizing raw data because of varying scales of measurements
raw.matrix <- scale(subset(testing, select = -c(quality, wine_is_good)))

# Additional experiments - which which is better for clustering?
# Repeating for more robust results (due to initialization)
# Using adjusted Rand index to compare similarity of labels (higher the better)
# 7 clusters for 7 unique quality scores (3 <= score <= 9)
cl.pe <- replicate(kmeans(strength.matrix, centers = 7)$cluster %>%
                     adjustedRandIndex(testing$quality), n = 1000)

# Now compare to clustering on raw data
cl.raw <- replicate(kmeans(raw.matrix, centers = 7)$cluster %>%
                      adjustedRandIndex(testing$quality), n = 1000)

results <- bind_rows(data.frame(adj.rand.index = cl.pe, method = "P.E."),
                     data.frame(adj.rand.index = cl.raw, method = "Raw"))

# Note that the clustering of prediction explanations yields a higher ARI
ggplot(results, aes(adj.rand.index, fill = method)) + geom_density(alpha = 0.2)

