import numpy as np
import pandas as pd
import gc

def old(x):
    np.log1p(np.maximum(x, 0))
    return None

def new(x):
    np.array(np.log1p(np.maximum(x, 0), dtype=np.float64), dtype=np.float32)
    return None

print('Downloading pandas')
dat = pd.read_csv('https://s3.amazonaws.com/datarobot_public_datasets/wild_function.csv', dtype=np.float32)

x = dat['x'].values
y = dat['y'].values
del dat

x -= x.min()
x /= x.max()

y -= y.min()
y /= y.max()

# x
@profile
def run_x():
    print('Running x')
    for _ in range(3): gc.collect()
    N = int(1e9)
    old(x[0:N])
    new(x[0:N])

    for _ in range(3): gc.collect()
    N = int(2e9)
    old(x[0:N])
    new(x[0:N])

    for _ in range(3): gc.collect()
    N = int(3e9)
    old(x[0:N])
    new(x[0:N])

    return None

# y
@profile
def run_y():
    print('Running y')
    for _ in range(3): gc.collect()
    N = int(1e9)
    old(y[0:N])
    new(y[0:N])

    N = int(2e9)
    for _ in range(3): gc.collect()
    old(y[0:N])
    new(y[0:N])

    N = int(3e9)
    for _ in range(3): gc.collect()
    old(y[0:N])
    new(y[0:N])

    return None

if __name__ == '__main__':
    run_x()
    run_y()
