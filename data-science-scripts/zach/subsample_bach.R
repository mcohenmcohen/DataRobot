library(data.table)
library(readr)
x <- fread('https://s3.amazonaws.com/datarobot_public_datasets/openML/datasets/BachChoralHarmony_v1.csv')
remap <- x[,list(.N), by='V17'][order(N),]
remap[,newname := V17]
remap[N <= 2, newname := 'other']
remap[,length(unique(newname))]
remap[,N := NULL]
x <- merge(x, remap, by='V17')
x[,V17 := newname]
x[,newname := NULL]
x[,length(unique(V17))]
write_csv(x, '~/datasets/BachChoralHarmony_v1_83_classes.csv')
