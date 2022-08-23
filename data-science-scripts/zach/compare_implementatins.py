######################################################################
# Setup
######################################################################
import numpy as np
from scipy.special import logit
from sklearn.utils.fixes import expit

def compare_funs(a, b, x):
  abs_diff = np.abs(a(x) - b(x))
  rel_diff = abs_diff / x
  print('max abs diff: ' + str(abs_diff.max()))
  print('max rel diff: ' + str(rel_diff.max()))

N = int(1e7)

######################################################################
# log1p / expm1
######################################################################
np.random.seed(531462)

small_uniform = np.random.uniform(size=N, low=0.0, high=1e-10)
med_uniform = np.random.uniform(size=N,low=0.0, high=100,)
large_uniform = np.random.uniform(size=N, low=0.0, high=1e10)

# Log1p
def manual_log1p(x):
  return np.log(x + 1)
print('Manual log1p, uniform small:')
compare_funs(np.log1p, manual_log1p, small_uniform)
print('Manual log1p, uniform med:')
compare_funs(np.log1p, manual_log1p, med_uniform)
print('Manual log1p, uniform large:')
compare_funs(np.log1p, manual_log1p, large_uniform)

# Expm1
def manual_expm1(x):
  return np.exp(x) - 1

large_uniform = np.random.uniform(size=N, low=0.0, high=200)
print('Manual expm1, uniform small:')
compare_funs(np.expm1, manual_expm1, np.log1p(small_uniform))
print('Manual expm1, uniform med:')
compare_funs(np.expm1, manual_expm1, np.log1p(med_uniform))
print('Manual expm1, uniform large:')
compare_funs(np.expm1, manual_expm1, np.log1p(large_uniform))

# float32 max
d = 'float32'
print(np.log(np.finfo(d).max))
upper_limit_pre_exp = np.array(88, dtype=d)
print(manual_expm1(upper_limit_pre_exp))
print(np.expm1(upper_limit_pre_exp))
compare_funs(np.expm1, manual_expm1, upper_limit_pre_exp)

######################################################################
# logit/expit
######################################################################

np.random.seed(481047)
uniform = np.random.uniform(size=N)
small_norm = np.random.normal(size=N, scale=1e-5)
med_norm = np.random.normal(size=N, scale=1)
large_norm = np.random.normal(size=N, scale=1e5)

# Logit
def manual_logit_1(x):
  return np.log(x) - np.log(1 - x)
def manual_logit_2(x):
  return np.log(x / (1 - x))

print('Manual logit 1, uniform:')
compare_funs(logit, manual_logit_1, uniform)
print('Manual logit 2, uniform:')
compare_funs(logit, manual_logit_2, uniform)

# Expit
def manual_expit(x):
  return 1/(1 + np.exp(-x))
  
print('Manual expit, uniform:')
compare_funs(expit, manual_expit, uniform)
print('Manual expit, small:')
compare_funs(expit, manual_expit, small_norm)
print('Manual expit, normal:')
compare_funs(expit, manual_expit, med_norm)
print('Manual expit, large:')
compare_funs(expit, manual_expit, large_norm)
