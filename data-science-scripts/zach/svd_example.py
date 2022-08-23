import numpy as np
from sklearn.decomposition import TruncatedSVD
rows = 100
X = np.ones((rows, 2))
svd = TruncatedSVD(n_components=2, random_state=1234)
svd.fit(X)
print(svd.components_)
print(np.round(svd.explained_variance_, 4))
print(np.round(svd.explained_variance_ratio_, 4))



# Make a 135 degree svd rSo here'otaiton matrix
true_svd_matrix = 1/np.sqrt(2) * np.array([[-1, 1], [1, -1]])
k = true_svd_matrix.shape[0]

# Make some data that rotates to this matrix
rows = 100
rs = np.random.RandomState(42)
X = rs.normal(size=(rows, k))
X = np.matmul(X, true_svd_matrix)

# Make the data wider
X = np.hstack([X, X])

# "true" svd
X = np.ones((rows, 2)) * 2
svd = TruncatedSVD(n_components=X.shape[1], random_state=42)
svd.fit(X)
print(svd.components_)
print(np.round(svd.explained_variance_, 4))
print(np.round(svd.explained_variance_ratio_, 4))

u, s, vh = svds(X, k=2)
print(np.round(vh, 2))
