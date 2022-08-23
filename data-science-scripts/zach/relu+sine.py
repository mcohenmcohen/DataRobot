import numpy as np
import matplotlib.pyplot as plt

def generate_piecewise_linear(n_samples = 500, periods=10, random_state = 123):
  rng = np.random.RandomState(random_state)

  X = np.linspace(0.0, periods, n_samples)

  # Piecewise linear basis function
  h0 = 1
  h1 = X
  h2 = np.maximum(0, X - 3)
  h3 = np.maximum(0, X - 6)

  y_true = (10 * h0) + (2 * h1) + (-1 * h2) + (1 * h3)

  return X, y_true

X, y_true = generate_piecewise_linear()
y_max = np.max(y_true)
y_min = np.min(y_true)
order = np.argsort(X)

plt.plot(X[order], (y_true[order] - y_min)/(y_max - y_min), label = 'x (normalized)')
plt.plot(X[order], np.sin(2 * np.pi * y_true[order]), label = 'sin(2 * pi * x)')
plt.legend()
plt.show()
