import numpy as np
from scipy.optimize import minimize

def mae_loss(X, y, coeff, w=None):
  pred = np.dot(X, coeff)
  if w is not None:
    return np.mean(np.abs(pred - y) * w)
  else:
    return np.mean(np.abs(pred - y))

def mae_regression(X, y, w=None):
  def wrapper(coeff):
    return np.array(mae_loss(X, y, coeff, w))
  x0 = np.zeros(X.shape[1])
  result = minimize(wrapper, x0, method='BFGS')
  return result.x

def mae_loss_L1(X, y, coeff, penalty, w=None):
  pred = np.dot(X, coeff)
  if w is not None:
    np.mean(np.abs(pred - y) * w) + np.mean(np.abs(coeff)) * penalty
  else:
    return np.mean(np.abs(pred - y)) + np.mean(np.abs(coeff)) * penalty

def mae_regression_L1(X, y, penalty, w=None):
  def wrapper(coeff):
    return np.array(mae_loss_L1(X, y, coeff, penalty, w))
  x0 = np.zeros(X.shape[1])
  result = minimize(wrapper, x0, method='BFGS')
  return result.x

rows = 1000
X = np.random.rand(1000, 5)
y = np.dot(X, np.array([100, 10, 1, 0, 0]))
c = np.zeros((X.shape[1], 1))

print(mae_loss(X, y, c))
print(np.round(mae_regression(X, y)))
print(np.round(mae_regression_L1(X, y, 2.4)))
