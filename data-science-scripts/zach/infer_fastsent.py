#!/usr/local/bin/python3
#http://www.cyberciti.biz/faq/python-command-line-arguments-argv-example/

import sys, getopt
from gensim.models.word2vec import LineSentence
from gensim.models import fastsent
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import logging

# Store input and output file names, number of vectors
ifile=''
ofile=''
modelfile = ''

# Read command line args
myopts, args = getopt.getopt(sys.argv[1:],"i:o:m:")

###############################
# o == option
# a == argument passed to the o
###############################
for o, a in myopts:
    if o == '-i':
        ifile=a
    elif o == '-o':
        ofile=a
    elif o == '-m':
        modelfile=a
    else:
        print("Usage: %s -i input -o output -m modelfile" % sys.argv[0])

###############################
# Load model and predict
###############################
model = fastsent.FastSent.load(modelfile)
print(
  model.sentence_similarity(
  "Obama speaks to the media in Illinois",
  "Hi you guys"
  )
)

#Sort vocab
#http://stackoverflow.com/questions/35960372/match-non-unique-un-sorted-array-to-indexes-in-unique-sorted-array
embedding = np.nan_to_num(model.syn0)
vocab = np.asarray(model.index2word)
sort_idx = np.argsort(vocab)
assert(len(sort_idx) == len(vocab))
vocab = vocab[sort_idx]

unmapped = -1
def match(x, table):
  out = np.searchsorted(table, x, 'left').astype('int32')
  right_idx = np.searchsorted(table, x, 'right')
  out[out == right_idx] = unmapped
  return(out)

#Infer embeddings
random.seed(2007)
x = pd.read_csv(ifile)[[0]]
x = x.replace(np.nan, ' ', regex=True)
x = x.values

#MUST BE THE SAME AS fastsent.FastSent ABOVE!
#implicitly removes missing tokens.  TODO: ADD ALL MISSING TOKENS BACK IN!!!!!
#TODO: lowercase missing tokens and try again
out = np.zeros((len(x), embedding.shape[1]), dtype="float64")
for i in range(len(x)):
  if i % 1000 == 0:
    print(i)
  tokens = x[i][0].split(" ")
  tokens = vocab[match(tokens, vocab)]
  sent = (" ").join([x for x in tokens])
  out[i,] = model[sent]

#Save the file
out_norm = normalize(out, 'l2')
np.save(ofile, out_norm)
