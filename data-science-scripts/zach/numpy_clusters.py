import numpy as np
import pandas as pd

rng = np.random.RandomState(42)
cluster_means = [(40, -60), (-10, -60), (50, 20), (-25, 25)]
cluster_cov = np.array([[1, 0], [0, 1]]).astype(np.float32)

out = []
for clust in cluster_means:
  out.append(rng.multivariate_normal(clust, cluster_cov * 10, 1000))
df = pd.DataFrame(np.vstack(out))
df.to_csv('~/workspace/data-science-scripts/zach/df.csv', index=False)
