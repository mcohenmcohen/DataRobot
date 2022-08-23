import numpy as np
import pandas as pd

N_ROWS = 10
RNG = np.random.default_rng(738047)

config = {
  "AcctID": (RNG.integers, {"low":0, "high": 2147483647, "size": N_ROWS}),
  
  "fake_target1": (RNG.binomial, {"n": 1, "p": 0.45, "size": N_ROWS}),
  "fake_target2": (RNG.binomial, {"n": 1, "p": 0.15, "size": N_ROWS}),
  "fake_target3": (RNG.binomial, {"n": 1, "p": 0.85, "size": N_ROWS}),
    
  "fake_target4": (RNG.binomial, {"n": 9, "p": 0.31, "size": N_ROWS}),
  "fake_target5": (RNG.binomial, {"n": 1, "p": 0.11, "size": N_ROWS}),
  "fake_target6": (RNG.binomial, {"n": 2, "p": 0.84, "size": N_ROWS}),
}
df = pd.DataFrame({k:v[0](**v[1]) for (k,v) in config.items()})
print(df)

