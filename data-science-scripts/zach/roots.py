from __future__ import division
import numpy as np
from functools import partial
from scipy.optimize import brent

x = np.random.normal(size=1000)
def func(x):
	return x / (1 + abs(x))
y = func(x)
del x

def inv_at_value(x, y):
	return y - func(x)

print brent(partial(inv_at_value, y=-10))
print brent(partial(inv_at_value, y=0))
print brent(partial(inv_at_value, y=10))
