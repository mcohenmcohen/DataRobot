# Requirements for https://github.com/pymc-devs/pymc3/blob/master/requirements.txt

# import theano
import numpy
import scipy
import pandas
import patsy
import joblib
# import tqdm
import six
import h5py
# import enum34

# print theano.__version__  # Need 1.0.0, do not have — NEED TO INSTALL
print numpy.__version__  # Need >=1.13.0, have 1.14.6.post1+dr - GOOD
print scipy.__version__  # Need >=0.18.1, 1.1.0.post4+dr- GOOD
print pandas.__version__  # Need >=0.23.0, have 0.23.0.post5+dr- GOOD
print patsy.__version__  # Need >=0.4.0, have 0.5.0- GOOD
print joblib.__version__  # Need >=0.9, have 0.8.0a4 - Close, might be good
# print tqdm.__version__  # Need >=4.8.4, do not have — NEED TO INSTALL
print six.__version__  # Need >=1.10.0, have 1.11.0 - GOOD
print h5py.__version__  # Need >=2.7.0, have 2.8.0 - GOOD
# print enum34.__version__  # Need >=1.1.6, do not have — NEED TO INSTALL


# Requirements for theano:
# https://github.com/Theano/Theano/blob/master/requirement-rtd.txt
import sphinx
import pygments
import nose
import numpy
# import gnumpy
# import pydot
# import pydot2
import Cython
# import parameterized
import scipy

print sphinx.__version__ # Need>=1.3.0, have 1.5.6 - GOOD
print pygments.__version__  # Have 2.1.1 - GOOD
print nose.__version__  # Need >=1.3.0, have 1.3.0 - GOOD
print numpy.__version__  # Have 1.14.6.post1+dr - GOOD
# print gnumpy.__version__  # DO NOT HAVE — NEED TO INSTALL
# print pydot.__version__  # DO NOT HAVE — NEED TO INSTALL
# print pydot2.__version__  # DO NOT HAVE — NEED TO INSTALL
print Cython.__version__  # Have 0.21 - GOOD
# print parameterized.__version__  # DO NOT HAVE — NEED TO INSTALL
print scipy.__version__  # Need ==0.13, have 1.1.0.post4+dr MAYBE NOT GOOD?
