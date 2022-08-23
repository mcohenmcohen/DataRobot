library(data.table)
library(maps)

two_models = fread('~/workspace/data-science-scripts/zach/lat_lon_example/2_model_preds.csv')
one_model = fread('~/workspace/data-science-scripts/zach/lat_lon_example/1_model_preds.csv')
cos_model = fread('~/workspace/data-science-scripts/zach/lat_lon_example/1_cos_model_preds.csv')

map('world')
points(cos_model[[2]], one_model[[1]], pch=19, col="red", cex=0.5)

