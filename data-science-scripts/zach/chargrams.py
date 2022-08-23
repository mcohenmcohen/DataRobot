# Based on https://github.com/minimaxir/char-embeddings
# https://github.com/minimaxir/char-embeddings/blob/master/text_generator_keras.py

from tqdm import tqdm
import numpy as np
import os

from keras.preprocessing.sequence import pad_sequences

file_path = "/Users/zachary/pretrained_nn/Text/glove.840B.300d.txt"

size = 2196017
vectors = {}
input_seq = size*[[0]]
output_mat = np.zeros(shape=(size, 300))

MIN = np.inf
MAX = MIN * -1
MAX_LEN = 0

with open(file_path, 'rb') as f:
    for i, line in enumerate(tqdm(f)):
        line_split = line.strip().split(b" ")
        vec = np.array(line_split[1:], dtype=float)
        word = line_split[0]
        charlist = list(word)

        if(len(charlist) <= 20):
          # if(len(charlist) > 20):
          #   print(word)

          chararray = np.array(charlist)

          MAX = max(MAX, chararray.max())
          MIN = min(MIN, chararray.min())
          MAX_LEN = max(MAX_LEN, len(charlist))

          input_seq[i] = charlist
          output_mat[i,:] = vec

          for char in word:
            char = chr(char)
            if char in vectors:
                vectors[char] = (vectors[char][0] + vec,
                                 vectors[char][1] + 1)
            else:
                vectors[char] = (vec, 1)

print((MAX, MIN, MAX_LEN, i))

#Save input and output for NN
input_mat = pad_sequences(input_seq)
assert input_mat.shape[0] == output_mat.shape[0]

#Save character glove vectors
base_name = os.path.splitext(os.path.basename(file_path))[0] + '-char.txt'
with open(base_name, 'wb') as f2:
    for word in vectors:
        avg_vector = np.round(
            (vectors[word][0] / vectors[word][1]), 6).tolist()
        f2.write(str(word) + " " + " ".join(str(x) for x in avg_vector) + "\n")

