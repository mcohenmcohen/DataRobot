
"""A multi-layer perceptron for neural decomposition."""
#https://github.com/HIPS/autograd/blob/master/examples/neural_net.py

from __future__ import absolute_import, division
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc.flatten import flatten

def layer_weights(n_in, n_out, scale, rs=npr.RandomState(42)):
    return [scale * rs.randn(n_in, n_out), scale * rs.randn(n_out)]

def layer_pred(inputs, W, b, activation):
    out = np.dot(inputs, W) + b
    return activation(out)

def init_random_params(scale, layer_sizes):
    """Build a list of (weights, biases) tuples,
       one for each layer in the net."""

    n_in = layer_sizes['input']
    n_out = layer_sizes['out']
    n_sin = layer_sizes['sin']
    n_total = sum(layer_sizes.values())
    n_mid = n_total - n_in - n_out
    out = {
        'sin': layer_weights(n_in, n_sin, scale),
        'out': layer_weights(n_mid, n_out, scale)
    }
    return out

def neural_net_predict(params, inputs):
    """Implements a deep neural network for classification.
       params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
       returns normalized class log-probabilities."""

    sin = layer_pred(inputs, params['sin'][0], params['sin'][1], np.sin)
    mid = np.hstack((sin, inputs))
    out = np.dot(mid, params['out'][0] + params['out'][1])
    return out

def l2_norm(params):
    """Computes l2 norm of params by flattening them into a vector."""
    flattened, _ = flatten(params)
    return np.dot(flattened, flattened)

def rmse(params, inputs, targets):
    error = targets - neural_net_predict(params, inputs)
    return np.sqrt(np.mean(error ** 2.))


X = np.arange(10).reshape(-1,1).astype(np.float32)
Y = 3*np.sin(10*X+2)

layer_sizes = {"input": 1, "sin": 64, "lin": 1, "out": 1}
L2_reg = 1.0
init_params = init_random_params(0.1, layer_sizes)

def objective(params):
    return rmse(params, X, Y)

objective_grad = grad(objective)

print(objective(init_params))
print(objective_grad(X))
