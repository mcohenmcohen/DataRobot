###############################################################
#
#   Keras utility functions
#
#   Author: Zach Deane-Mayer
#
#   Copyright DataRobot Inc, 2018 onwards
#
###############################################################
from __future__ import division, print_function

import os
import json
import logging
from copy import deepcopy

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from scipy.special import expit  # pylint: disable=no-name-in-module
from scipy.special import logit  # pylint: disable=no-name-in-module
from scipy.stats import norm

from common.engine.parallel import get_n_jobs
from common.exceptions import TaskError


# Logger
logger = logging.getLogger("datarobot")

#############################################################################
# Constants
#############################################################################

# Integer min/max
# From np.iinfo(np.int32), +1 to min -1 to max
# Used for drawing random seeds
IMIN = -2147483647
IMAX = 2147483646

# Float32 min/max
# Real max is -3.4028235e38, but leave some wiggle room
FMIN = -1e+38
FMAX = 1e+38

# Initializers
KERAS_INITS = ['zeros', 'ones', 'random_uniform', 'lecun_uniform',
               'glorot_uniform', 'he_uniform', 'random_normal',
               'lecun_normal', 'glorot_normal', 'he_normal',
               'truncated_normal', 'VarianceScaling', 'orthogonal']

# Activations
KERAS_ACTIVATIONS = ['linear', 'sigmoid', 'hard_sigmoid', 'relu', 'elu', 'selu',
                     'tanh', 'softmax', 'softplus', 'softsign', 'exponential',
                     'swish', 'thresholdedrelu', 'leakyrelu', 'prelu', 'cloglog', 'probit']
KERAS_ADVANCED_ACTIVATIONS = ['leakyrelu', 'prelu', 'thresholdedrelu']
KERAS_EXPONENTIAL_ACTIVATIONS = ['exponential', 'selu', 'elu']
KERAS_NEGONE_ONE_ACTIVATIONS = ['tanh', 'softsign']
KERAS_OUTPUT_ACTIVATIONS = ['linear', 'sigmoid', 'softsign', 'exponential', 'tanh', 'cloglog',
                            'softplus', 'probit', 'softmax', 'selu', 'elu']

# Losses
# Note: categorical_hinge and categorical_crossentropy intentionally omitted.
# Those 2 losses require 1-hot-encoded targets, which we do not do
# hinge (for 0/1 targets) and sparse_categorical_crossentropy (for integer targets)
# Are supported instead
# https://keras.io/losses
# 'sparse_categorical_crossentropy',  # Won't work for class/reg models.  Only for multiclass
KERAS_LOSSES = ['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error',
                'mean_squared_logarithmic_error', 'squared_hinge', 'hinge', 'logcosh',
                'binary_crossentropy', 'kullback_leibler_divergence', 'poisson', 'gamma', 'tweedie',
                'cosine_proximity']
KERAS_LOSSES_HIGHER_IS_BETTER = ['cosine_proximity']
KERAS_LOSSES_LOWER_IS_BETTER = list(set(KERAS_LOSSES) - set(KERAS_LOSSES_HIGHER_IS_BETTER))

# Optimizers
KERAS_OPTIMIZERS = ['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adamax', 'nadam']


#############################################################################
# Logger helper
#############################################################################


def log_error_and_raise_to_user(err, extra):
    """Log an error to DataRobot's logs and raise it to the user"""
    logger.error(err, extra=extra)


#############################################################################
# Utility functions for working with tensorflow/keras
#############################################################################

def get_initializer(name, seed):
    """Get a given initializer from Keras"""
    if name in ['zeros', 'ones']:
        return keras.initializers.get(name)
    else:
        return keras.initializers.get({'class_name': name, 'config': {'seed': seed}})


def get_regularizer(l1_val, l2_val):
    """ Create Keras regularizer. """
    if l1_val > 0 and l2_val > 0:
        return keras.regularizers.l1_l2(l1_val, l2_val)
    elif l1_val > 0:
        return keras.regularizers.l1(l1_val)
    elif l2_val > 0:
        return keras.regularizers.l2(l2_val)
    else:
        return None


def get_model_memory_usage(model, batch_size, dtype):
    """
    Estimate memory of a keras model
    Code from Sabari
    See also https://stackoverflow.com/a/46216013

    :param model: A keras model
    :param batch_size: An integer batch size
    :param dtype: The dtype you will use (usually float32)
    :return: A tuple: (ram to train the model, size of the pickle) both estimates
    """
    # Map dtype to size
    dtype_size = 4.0
    if dtype == 'float16':
        dtype_size = 2.0
    if dtype == 'float64':
        dtype_size = 8.0

    # Count Shapes
    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for out_s in l.output_shape:
            if out_s:
                single_layer_mem *= out_s
        shapes_mem_count += single_layer_mem

    # Count parameters
    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])
    total_memory = dtype_size * (
        batch_size * shapes_mem_count + trainable_count + non_trainable_count)

    # Training memory usage (for training on a batch)
    ram_gbytes = total_memory / (1024.0 ** 3)

    # Vertex Model size (for pickling)
    total_memory = dtype_size * (trainable_count + non_trainable_count)
    pickle_gbytes = total_memory / (1024.0 ** 3)

    return ram_gbytes, pickle_gbytes


def validate_approx_model_size(model, batch_size, dtype):
    """Validates that DataRobot will have enough RAM to train a given model"""

    # Estimate RAM usage to train and model size as a pickle:
    ram_gbytes, pickle_gbytes = get_model_memory_usage(model, batch_size, dtype)

    # Warn if pickle is likely to be > 2GB
    # Since we're on python 2, we can't pickle objects > 2GB
    # batch size of 1 is the size of the model itself, on 1 row of data
    if pickle_gbytes > 2:
        logger.warn('Model is likely too big to pickle',
                    extra={'Estimated Pickle GB': np.round(pickle_gbytes, 1)})

    # If pickle > 4 GB, error, as this will def fail.
    if pickle_gbytes > 4:
        log_error_and_raise_to_user(
            'Model is too big to pickle. Please reduce the number of layers and number of hidden '
            'units.  Also consider tuning the preprocessing for fewer inputs.',
            extra={'Estimated Pickle GB': np.round(ram_gbytes, 1)}
        )

    # One row of data ~= the model size after
    # >50GB here is very likely to cause memory issues
    # Consider increasing for 100GB autopilot project
    # Assume 10GB of data + 50GB of Model = 60GB of RAM ib our yarn containers
    if ram_gbytes > 50:
        log_error_and_raise_to_user(
            'Model is too large.  Please reduce the batch size (or number of units/layers).',
            extra={'Estimated RAM Usage GB': np.round(ram_gbytes, 1)}
        )


def calculate_lr_decay(initial_lr, final_lr, n_batches):
    return np.power(final_lr / initial_lr, 1 / n_batches)


def pretty_print_model_json(x):
    """Print a nice representation of the model's JSON structure"""
    out = json.loads(x.model.to_json())
    print(json.dumps(out, indent=4, sort_keys=True))


#############################################################################
# Numpy versions of activations / inverse activations
#############################################################################


def elu(x):
    out = deepcopy(x)
    out = np.expm1(out, out=out, where=out < 0)
    return out


def inv_elu(x):
    out = deepcopy(x)
    return np.log1p(out, out=out, where=out < 0)


def selu(x):
    out = deepcopy(x)
    out = np.expm1(out, out=out, where=out < 0)
    out = np.multiply(out, 1.67326, out=out, where=out < 0)
    return 1.0507 * out


def inv_selu(x):
    out = x / 1.0507
    out = np.divide(out, 1.67326, out=out, where=out < 0)
    out = np.log1p(out, out=out, where=out < 0)
    return out


def softplus(x):
    return np.log1p(np.exp(x))


def inv_softplus(x):
    return np.log(np.expm1(x))


def softsign(x):
    return x / (1 + np.abs(x))


def inv_softsign(x):
    out1 = -x / (x - 1)
    out2 = x / (x + 1)
    out = out1
    idx = np.where(x < 0)
    out[idx] = out2[idx]
    return out


# http://www.spiderfinancial.com/support/documentation/numxl/reference-manual/transform/cloglog
def cloglog(x):
    return 1 - np.exp(-np.exp(x))


def inv_cloglog(x):
    return np.log(-np.log(1 - x))


numpy_activations = {
    'linear': lambda x: x,
    'sigmoid': expit,
    'elu': elu,
    'selu': selu,
    'tanh': np.tanh,
    'softmax': expit,  # Not sure on this, should work
    'softplus': softplus,
    'softsign': softsign,
    'exponential': np.exp,
    'probit': norm.cdf,
    'cloglog': cloglog,
}

numpy_inverse_activations = {
    'linear': lambda x: x,
    'sigmoid': logit,
    'elu': inv_elu,
    'selu': inv_selu,
    'tanh': np.arctanh,
    'softmax': logit,  # Not sure on this, should work
    'softplus': inv_softplus,
    'softsign': inv_softsign,
    'exponential': np.log,
    'probit': norm.ppf,  # https://stackoverflow.com/a/20627638
    'cloglog': inv_cloglog,
}


#############################################################################
# Custom loss functions and activations
#############################################################################


# https://github.com/datarobot/DataRobot/blob/master/ModelingMachine/engine/metrics/__init__.py
# #L1077
def gamma_loss(act, pred, weight=None):
    """Gamma deviance"""
    pred = K.maximum(pred, 1e-12)
    d = 2 * (-K.log(act / pred) + (act - pred) / pred)
    if weight is not None:
        d = d * weight / K.mean(weight)
    return K.mean(d)


# https://github.com/datarobot/DataRobot/blob/master/ModelingMachine/engine/metrics/__init__.py
# #L1009
def tweedie_loss(act, pred, weight=None, p=1.5):
    """tweedie deviance for p = 1.5 only"""
    pred = K.maximum(pred, 1e-12)
    if p <= 1 or p >= 2:
        raise ValueError('p equal to %s is not supported' % p)

    d = ((act ** (2.0 - p)) / ((1 - p) * (2 - p)) -
         (act * (pred ** (1 - p))) / (1 - p) + (pred ** (2 - p)) / (2 - p))
    d = 2 * d
    if weight is not None:
        d = d * weight / K.mean(weight)
    return K.mean(d)


def swish_activation(x):
    """Some paper said this was good"""
    return K.sigmoid(x) * x


def cloglog_activation(x):
    """
    complementary log-log link function for binary models
    equivalent to -np.expm1(-np.exp(x))
    """
    return -(K.exp(-K.exp(x)) - 1)


def probit_activation(x):
    normal = tf.distributions.Normal(loc=0., scale=1.)
    return normal.cdf(x)


def register_custom_keras_functions():
    class NamedActivation(keras.layers.Activation):
        def __init__(self, activation_function, name, **kwargs):
            super(NamedActivation, self).__init__(activation_function, **kwargs)
            self.__name__ = name

    get_custom_objects = keras.utils.generic_utils.get_custom_objects
    get_custom_objects().update({'gamma': gamma_loss})
    get_custom_objects().update({'tweedie': tweedie_loss})
    get_custom_objects().update({'swish': NamedActivation(swish_activation, 'swish')})
    get_custom_objects().update({'cloglog': NamedActivation(cloglog_activation, 'cloglog')})
    get_custom_objects().update({'probit': NamedActivation(probit_activation, 'probit')})


# Activations
def get_advanced_actications(x):
    ADVANCED_ACTIVATIONS_DICT = {
        'leakyrelu': keras.layers.LeakyReLU,
        'prelu': keras.layers.PReLU,
        'thresholdedrelu': keras.layers.ThresholdedReLU,
    }
    return ADVANCED_ACTIVATIONS_DICT.get(x)


def get_activation(act_string):
    """Helper function for working with advanced activations.  """
    if act_string in KERAS_ADVANCED_ACTIVATIONS:
        return {'act_string': 'linear', 'advanced_act': get_advanced_actications(act_string)}
    else:
        return {'act_string': act_string, 'advanced_act': None}
