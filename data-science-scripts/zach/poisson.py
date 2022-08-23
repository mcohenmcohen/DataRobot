# Data
import numpy as np
s = np.random.seed(42)
s = np.random.poisson(0.5, 150)
assert len(np.unique(s)) <= 10 # Multiclass must be <= 10 classses

# Checks
# Based on looks_like_poisson from
# ModelingMachine/engine/eda_multi.py
import scipy.stats as stats
assert s.min() == 0
assert stats.scoreatpercentile(s, 85) != s.max()
assert s.var() < 50 * s.mean()
assert stats.skew(s) > 0.5

#Plot
import matplotlib.pyplot as plt
count, bins, ignored = plt.hist(s, 14, normed=True)
plt.show()
