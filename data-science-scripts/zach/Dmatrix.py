import xgboost as xgb
from scipy.sparse import random
csc = random(100, 10, format='csc')
dtrain = xgb.DMatrix(csc, missing=0)
