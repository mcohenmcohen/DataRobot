library(data.table)
x=fread('https://s3.amazonaws.com/datarobot_public_datasets/MusicListeningHistories_1GB_user_metadata.csv')
fwrite(x, '~/datasets/MusicListeningHistories_1GB_user_metadata_cleaned.csv')
