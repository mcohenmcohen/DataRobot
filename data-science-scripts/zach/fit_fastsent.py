#!/usr/local/bin/python3
#http://www.cyberciti.biz/faq/python-command-line-arguments-argv-example/

import sys, getopt
from gensim.models.word2vec import LineSentence
from gensim.models import fastsent
import random
import logging

# Store input and output file names, number of vectors
ifile=''
ofile=''
N_VEC = 100

# Read command line args
myopts, args = getopt.getopt(sys.argv[1:],"i:o:n:")

###############################
# o == option
# a == argument passed to the o
###############################
for o, a in myopts:
    if o == '-i':
        ifile=a
    elif o == '-o':
        ofile=a
    elif o == '-n':
        N_VEC=a
    else:
        print("Usage: %s -i input -o output -n number" % sys.argv[0])

###############################
# Fit model
###############################

random.seed(9515)
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = LineSentence(ifile)
model = fastsent.FastSent(sentences, min_count=10, workers=4, size=N_VEC, window=5, iter=10, sample=1e-4, autoencode=True)
print(model.sentence_similarity('Sunday is a computer virus, a member of the Jerusalem virus family', 'It was discovered in November 1989'))

###############################
# Save model
###############################
model.save(ofile)
