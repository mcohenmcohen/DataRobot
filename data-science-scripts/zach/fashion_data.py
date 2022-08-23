#!/usr/bin/python

# Setup
from PIL import Image
import os, sys
from tqdm import tqdm
import pandas as pd
import ujson

# Data from https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset
# Some code ideas from #https://stackoverflow.com/a/21518989

# Resize and organize all the images
# About 45 mins
# Loop through JSON
# Select apparel only for now
# Select stuff that is >$100 only for now
# Organize images into folders by number of likes
in_path_images = "/Users/zachary/Downloads/fashion-dataset/images/"
in_path_json = "/Users/zachary/Downloads/fashion-dataset/styles/"
out_path = "/Users/zachary/Downloads/fashion-dataset/output-dataset/"
dirs = os.listdir(in_path_json)
for item in tqdm(dirs):
  
  with open(in_path_json + item) as f:
    data = ujson.load(f)
  
  if data['data']['masterCategory']['typeName'] == 'Apparel':
    if data['data']['price'] > 1000:
      likes = data['data']['articleDisplayAttr']['social']['userLikes']
      out_dir = out_path + likes + '/'
      if not os.path.exists(out_dir):
        os.makedirs(out_dir)
      
      item_trimmed, extension = os.path.splitext(item)
      im = Image.open(in_path_images + item_trimmed + '.jpg')

      imResize = im.resize((224, 224), Image.ANTIALIAS)
      imResize.save(out_dir + item_trimmed + '.jpg', 'JPEG', quality=90)

