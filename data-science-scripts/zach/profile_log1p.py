import cProfile
import numpy as np
import pandas as pd
import gc

def old(x):
    return np.log1p(np.maximum(x, 0))

def new(x):
    return np.array(np.log1p(np.maximum(x, 0), dtype=np.float64), dtype=np.float32)

dat = pd.read_csv('https://s3.amazonaws.com/datarobot_public_datasets/wild_function.csv', dtype=np.float32)

x = dat['x'].values
y = dat['y'].values
del dat

x -= x.min()
x /= x.max()

y -= y.min()
y /= y.max()

# 1 million - 3 billion
for N in [1e6, 1e7, 1e8, 1e9, 2e9, 3e9]:
    N = int(N)
    print(str(N))
    for F in ['old', 'new']:
        print('...' + F)
        for _ in range(3): gc.collect()
        cProfile.run('{F}(y[0:{N}])'.format(F=F, N=N))

# 1 million - 3 billion
for N in [1e6, 1e7, 1e8, 1e9, 2e9, 3e9]:
    N = int(N)
    print(str(N))
    for F in ['old', 'new']:
        print('...' + F)
        for _ in range(3): gc.collect()
        cProfile.run('{F}(y[0:{N}])'.format(F=F, N=N))
