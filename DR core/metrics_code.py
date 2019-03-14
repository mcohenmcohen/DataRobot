# pylint: disable=unused-argument
"""@package ModelingMachine.engine.metrics prediction scoring methods
"""
######################################################
#
#   Metrics
#
#   Author: Tom DeGodoy
#
#   Copyright DataRobot Inc, 2013 onwards
#
######################################################

from __future__ import division
import itertools
# used for METRIC_MAP serialization
import json
import os
import copy
import hashlib

import scipy as sp
import numpy as np
import pandas as pd
import logging
from math import exp

from collections import defaultdict

# pylint: disable=wildcard-import
from common.engine.metrics import *
from ModelingMachine.engine.container import Container
from ModelingMachine.engine.metrics.match_funcs import *
from ModelingMachine.engine.metrics.wrappers import metricify, METRIC_MAP
from common.validation import accepts
from common.engine import InputType
from common.utilities.array_utils import probabilities_to_classes


from sklearn.metrics import roc_auc_score as auc_score, accuracy_score, recall_score, roc_curve
from tesla.utils import check_array

from numpy.random import RandomState

logger = logging.getLogger('datarobot')

DIRECTION_DESCENDING = -1
DIRECTION_ASCENDING = 1


# ----------------------------------------------------------------------------------------------
#
#     Binary Classification Metrics
#
# ----------------------------------------------------------------------------------------------


def logloss1D(act, pred, weight=None):
    """ Vectorized computation of logloss """
    if len(pred.shape) > 1:
        pred = pred.ravel()

    # convert to float64 for common precision between models.
    # this will trigger a copy for float32 input
    pred = check_array(pred, dtype=np.float64, copy=False)

    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)

    d = - (act * sp.log(pred) + sp.subtract(1, act) * sp.log(sp.subtract(1, pred)))
    if weight is not None:
        if weight.sum() == 0:
            return 0
        d = d * weight / weight.mean()
    return d.mean()


def logloss2D(act, pred, weight=None):
    # convert to float64 for common precision between models.
    # this will trigger a copy for float32 input
    pred = check_array(pred, dtype=np.float64, copy=False)

    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)

    binary_actual = np.empty(pred.shape, dtype=np.float64)
    for i in range(pred.shape[1]):
        binary_actual[:, i] = act == i
    d = -sp.sum(sp.multiply(binary_actual, sp.log(pred)), axis=1)
    if weight is not None:
        if weight.sum() == 0:
            return 0
        d = d * weight / weight.mean()
    return d.mean()

def _logloss(act, pred, weight=None):
    if len(pred.shape) == 2 and pred.shape[1] == 1:
        pred = pred.ravel()
    if len(pred.shape) == 1:
        return logloss1D(act, pred, weight)
    else:
        return logloss2D(act, pred, weight)


def _average_multiclass_auc(y_true, y_pred, sample_weight=None):
    """ Multi-class AUC as the average of binary AUC scores """
    score = 0
    for i in range(y_pred.shape[1]):
        binary_target = y_true == i
        if sample_weight is not None:
            weight = np.sum(sample_weight * binary_target) / np.sum(sample_weight)
        else:
            weight = np.sum(binary_target) / y_pred.shape[0]
        if len(np.unique(binary_target)) > 1:
            score += weight * auc_score(binary_target, y_pred[:, i], sample_weight=sample_weight)
        else:
            # For binary we return AUC=1 when only one class present, do
            # the same in multiclass case for consistency (weight * 1)
            score += weight
    return score


@metricify(LOGLOSS, "Measures Inaccuracy of Predicted Probabilities",
           DIRECTION_ASCENDING, match_func=match_logloss,
           weighted_version=LOGLOSS_W)
def logloss(act, pred, weight=None):
    return _logloss(act, pred, weight)


@metricify(LOGLOSS_W, "Measures Inaccuracy of Predicted Probabilities",
           DIRECTION_ASCENDING, match_func=match_logloss_w)
def logloss_w(act, pred, weight=None):
    return _logloss(act, pred, weight)


def _accuracy(act, pred, weight=None):
    if pred.ndim == 2:
        pred = probabilities_to_classes(pred)
    return accuracy_score(y_true=np.rint(act),
                          y_pred=np.rint(pred),
                          normalize=True,
                          sample_weight=weight)


@metricify(ACCURACY, "Fraction of correctly classified samples",
           DIRECTION_DESCENDING, match_func=match_accuracy,
           weighted_version=ACCURACY_W)
def accuracy(act, pred, weight=None):
    return _accuracy(act, pred, weight)


@metricify(ACCURACY_W, "Weighted fraction of correctly classified samples",
           DIRECTION_DESCENDING, match_func=match_accuracy_w)
def accuracy_w(act, pred, weight=None):
    return _accuracy(act, pred, weight)


def _balanced_accuracy(act, pred, weight=None):
    if pred.ndim == 2:
        pred = probabilities_to_classes(pred)
    return recall_score(y_true=np.rint(act),
                        y_pred=np.rint(pred),
                        pos_label=None,
                        average='macro',
                        sample_weight=weight)


@metricify(BALANCED_ACCURACY, "Average accuracy per target class",
           DIRECTION_DESCENDING, match_func=match_balanced_accuracy,
           weighted_version=BALANCED_ACCURACY_W)
def balanced_accuracy(act, pred, weight=None):
    return _balanced_accuracy(act, pred, weight)


@metricify(BALANCED_ACCURACY_W, "Average weighted accuracy per target class",
           DIRECTION_DESCENDING, match_func=match_balanced_accuracy_w)
def balanced_accuracy_w(act, pred, weight=None):
    return _balanced_accuracy(act, pred, weight)


def _fve_binomial(act, pred, weight=None):
    v = _logloss(act, w_mean(act, weight), weight)
    if v > 0:
        return (1 - _logloss(act, pred, weight) / v)
    else:
        return 0


@metricify(FVE_BINOMIAL, "Fraction of variance explained for binomial deviance",
           DIRECTION_DESCENDING, match_func=match_fve_binomial,
           weighted_version=FVE_BINOMIAL_W)
def fve_binomial(act, pred, weight=None):
    return _fve_binomial(act, pred, weight)


@metricify(FVE_BINOMIAL_W, "Fraction of variance explained for binomial deviance",
           DIRECTION_DESCENDING, match_func=match_fve_binomial_w)
def fve_binomial_w(act, pred, weight=None):
    return _fve_binomial(act, pred, weight)


# --------------------------------------------------------------------
def _logloss_sig(act, pred, weight=None):
    ''' apply sigmoid to args before calculating log loss.
        used for model approximation of probabilities '''
    sa = 1 / (1 + np.exp(-act))
    sp = 1 / (1 + np.exp(-pred))
    return _logloss(sa, sp, weight)


@metricify(LOGLOSS_SIG, "For internal use only",
           DIRECTION_ASCENDING, match_func=match_disabled,
           weighted_version=LOGLOSS_SIG_W)
def logloss_sig(act, pred, weight=None):
    ''' apply sigmoid to args before calculating log loss.
        used for model approximation of probabilities '''
    return _logloss_sig(act, pred, weight)


@metricify(LOGLOSS_SIG_W, "For internal use only",
           DIRECTION_ASCENDING, match_func=match_disabled)
def logloss_sig_w(act, pred, weight=None):
    ''' apply sigmoid to args before calculating log loss.
        used for model approximation of probabilities '''
    return _logloss_sig(act, pred, weight)


# --------------------------------------------------------------------
def _auc(act, pred, weight=None):
    if len(np.unique(act)) == 1:
        return 1
    if len(np.unique(pred)) == 1:
        return 0.5
    if len(pred.shape) == 1:
        if weight is None:
            return auc_score(act, pred)
        else:
            return auc_score(act, pred, sample_weight=weight)
    else:
        out = []
        for col in range(pred.shape[1]):
            p = pred[:, col]
            if weight is None:
                out.append(auc_score(act, p))
            else:
                out.append(auc_score(act, p, sample_weight=weight))
        return max(out)


@metricify(AUC, "Measures ability to Distinguish the Ones from the Zeros",
           DIRECTION_DESCENDING, match_func=match_auc, ranking=True,
           weighted_version=AUC_W)
def auc(act, pred, weight=None):
    if pred.ndim > 1 and pred.shape[1] > 2:
        return _average_multiclass_auc(act, pred, weight)
    else:
        return _auc(act, pred, weight)


@metricify(AUC_W, "Measures ability to Distinguish the Ones from the Zeros",
           DIRECTION_DESCENDING, match_func=match_auc_w, ranking=True)
def auc_w(act, pred, weight=None):
    if pred.ndim > 1 and pred.shape[1] > 2:
        return _average_multiclass_auc(act, pred, weight)
    else:
        return _auc(act, pred, weight)


# --------------------------------------------------------------------
def _ks_score(act, pred, weight=None):
    fpr, tpr, _ = roc_curve(act, pred, sample_weight=weight)
    return max(tpr - fpr)


def _ks(act, pred, weight=None):
    if len(np.unique(act)) == 1:
        return 1.0
    if len(np.unique(pred)) == 1:
        return 0.0
    if len(pred.shape) == 1:
        return _ks_score(act, pred, weight)
    elif pred.shape[1] == 1:
        return _ks_score(act, pred.ravel(), weight)
    elif pred.shape[1] == 2:
        out = []
        for col in range(pred.shape[1]):
            p = pred[:, col]
            out.append(_ks_score(act, p, weight))
        return max(out)
    else:
        raise ValueError('Kolmogorov-Smirnov does not support multiclass prediction')


@metricify(KS, "Measures the degree of separation between the positive "
               "and negative distributions",
           DIRECTION_DESCENDING, match_func=match_ks, ranking=True)
def ks(act, pred, weight=None):
    return _ks(act, pred, weight)


@metricify(KS_W, "Measures the degree of separation between the positive "
                 "and negative distributions",
           DIRECTION_DESCENDING, match_func=match_ks_w, ranking=True)
def ks_w(act, pred, weight=None):
    return _ks(act, pred, weight)


# --------------------------------------------------------------------
def gini_score(act, pred, weight=None):
    act = np.asarray(act)
    pred = np.asarray(pred)
    assert (act.shape[0] == pred.shape[0])
    if np.any(act < 0):
        act = act - act.min()

    # If all predictions are the same then gini_score = 0
    if len(np.unique(pred)) == 1:
        return 0.0

    # This shuffles the rows to avoid leaks when the data is sorted by
    # target and predictions have equal values
    act_perm = RandomState(42).permutation(np.arange(len(act)))

    if weight is not None:
        # version from Kaggle
        # https://www.kaggle.com/c/liberty-mutual-fire-peril/forums/t/9880/update-on-the-evaluation-metric
        assert (act.shape[0] == weight.shape[0])
        all_data = np.asarray(np.c_[act, pred, act_perm, weight],
                              dtype=np.float)
        all_data = all_data[np.lexsort((all_data[:, 2], -1 * all_data[:, 1]))]
        ra = all_data[:, 3].cumsum() / all_data[:, 3].sum()
        Lo = (all_data[:, 0] * all_data[:, 3]).cumsum() / \
            (all_data[:, 0] * all_data[:, 3]).sum()
        n = len(all_data[:, 0])
        res = (Lo[0:(n - 1)] * ra[1:n]).sum() - (Lo[1:n] * ra[0:(n - 1)]).sum()
    else:
        # version from Kaggle
        # http://www.kaggle.com/c/ClaimPredictionChallenge/forums/t/703/code-to-calculate-normalizedgini
        all_data = np.asarray(np.c_[act, pred, act_perm], dtype=np.float)
        all_data = all_data[np.lexsort((all_data[:, 2], -1 * all_data[:, 1]))]
        totalLosses = all_data[:, 0].sum()
        if totalLosses == 0:
            return 1
        giniSum = all_data[:, 0].cumsum().sum() / totalLosses
        giniSum -= (len(act) + 1) / 2.
        # res = giniSum / len(act)*2 # SY add *2
        res = giniSum / len(act)  # SY keep without 2 for now
        # another Kaggle version
        # https://www.kaggle.com/wiki/RCodeForGini
        # all = np.asarray(np.c_[ act, pred, np.arange(len(act)) ], dtype=np.float)
        # all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
        # ra=(np.arange(len(act))+1)/len(act)
        # tPo=all[:,0].sum()
        # cPoFo=all[:,0].cumsum()
        # Lo=cPoFo/tPo
        # Gi=Lo-ra
        # res=Gi.sum()/len(act)*2

    if not np.isfinite(res):
        logger.warning('gini_score is not finite: %r', res, extra={'data': {
            'len(act)': len(act),
        }})
        res = 0.0
    return res


def _gini(act, pred, weight=None):
    if len(pred.shape) == 1:
        return gini_score(act, pred, weight)
    else:
        out = []
        for col in range(pred.shape[1]):
            p = pred[:, col]
            out.append(gini_score(act, p, weight))
        return max(out)


@metricify(GINI, "Measures Ability to Rank",
           DIRECTION_DESCENDING, ranking=True, match_func=match_gini,
           weighted_version=GINI_W)
def gini(act, pred, weight=None):
    return _gini(act, pred, weight)


@metricify(GINI_W, "Measures Ability to Rank",
           DIRECTION_DESCENDING, ranking=True, match_func=match_gini_w)
def gini_w(act, pred, weight=None):
    return _gini(act, pred, weight)


# --------------------------------------------------------------------
def _gini_norm(act, pred, weight=None):
    # for binary use sklearn auc
    if set(np.unique(act)) == set([0, 1]):
        return auc_w(act, pred, weight) * 2.0 - 1.0
    act_gini = _gini(act, act, weight)
    if act_gini == 0.0:
        return 0.0
    else:
        pred_gini = _gini(act, pred, weight)
        return pred_gini / act_gini


@metricify(GINI_NORM, "Measures Ability to Rank",
           DIRECTION_DESCENDING, match_func=match_gini_norm, ranking=True,
           weighted_version=GINI_NORM_W)
def gini_norm(act, pred, weight=None):
    return _gini_norm(act, pred, weight)


@metricify(GINI_NORM_W, "Measures Ability to Rank",
           DIRECTION_DESCENDING, match_func=match_gini_norm_w, ranking=True)
def gini_norm_w(act, pred, weight=None):
    return _gini_norm(act, pred, weight)


# --------------------------------------------------------------------
def ians_metric_1d(act, pred, weight=None):
    # weighted version - ?
    m = pd.Series(pred).groupby(pd.Series(act)).mean().to_dict()
    return m[1] - m[0]


@metricify(IANS_METRIC, "Ian's Metric",
           DIRECTION_DESCENDING, match_func=match_disabled)
def ians_metric(act, pred, weight=None):
    if len(np.unique(act)) == 1:
        return 1
    if len(pred.shape) == 1:
        return ians_metric_1d(act, pred)
    else:
        out = []
        for col in range(pred.shape[1]):
            p = pred[:, col]
            out.append(ians_metric_1d(act, p))
        return max(out)


# --------------------------------------------------------------------
def weighted_percentile_1d(vals, weights, percentile):
    """
    Compute the weighted percentile of a 1D numpy array.

    Parameters
    ----------
    vals : ndarray
        Input array (one dimension).
    weights : ndarray
        Array with the weights of the same size of `data`.
    percentile : float
        Percentile to compute. It must have a value between 0 and 100.

    Returns
    -------
    float
    """
    if len(vals.shape) > 1:
        vals = vals.ravel()
    if len(weights.shape) > 1:
        weights = weights.ravel()

    if vals.shape != weights.shape:
        raise TypeError("the shape of vals and weights are different")
    if ((percentile > 100.) or (percentile < 0.)):
        raise ValueError("percentile must be between 0. and 100.")

    # Sort values and weights
    ind_sorted = np.argsort(vals)
    sorted_vals = vals[ind_sorted]
    sorted_weights = weights[ind_sorted]

    # Compute the observed discrete percentiles
    percentiles = np.cumsum(sorted_weights) / np.sum(sorted_weights)

    # interpolate to get desired percentile
    return np.interp(float(percentile)/100, percentiles, sorted_vals)


# --------------------------------------------------------------------
def rate_top10_1d(act, pred, weight=None):
    """
        response rate in top 10% highest predictions
    """
    if len(pred.shape) > 1:
        pred = pred.ravel()

    if weight is None:
        top = (pred >= np.percentile(pred, 90))
    else:
        # get top 10% based on weighted percentile
        if len(weight.shape) > 1:
            weight = weight.ravel()
        top = (pred >= weighted_percentile_1d(pred, weight, 90))
        # compute weighted rate
        top = top.astype(np.float32) * weight

    if top.sum() == 0:
        return 0
    return (act * top).sum() / top.sum()


def _rate_top10(act, pred, weight=None):
    if len(pred.shape) == 1:
        return rate_top10_1d(act, pred, weight)
    else:
        out = []
        for col in range(pred.shape[1]):
            p = pred[:, col]
            out.append(rate_top10_1d(act, p, weight))
        return max(out)


@metricify(RATE_TOP10_PCT, "Response rate in top 10% highest predictions",
           DIRECTION_DESCENDING, match_func=match_binary)
def rate_top10(act, pred, weight=None):
    return _rate_top10(act, pred, weight)


@metricify(RATE_TOP10_PCT_W, "Response rate in top 10% highest predictions",
           DIRECTION_DESCENDING, match_func=match_binary)
def rate_top10_w(act, pred, weight=None):
    return _rate_top10(act, pred, weight)


# --------------------------------------------------------------------
def rate_top05_1d(act, pred, weight=None):
    """
        response rate in top 5% highest predictions
    """
    if len(pred.shape) > 1:
        pred = pred.ravel()

    if weight is None:
        top = (pred >= np.percentile(pred, 95))
    else:
        # get top 10% based on weighted percentile
        if len(weight.shape) > 1:
            weight = weight.ravel()
        top = (pred >= weighted_percentile_1d(pred, weight, 95))
        # compute weighted rate
        top = top.astype(np.float32) * weight

    if top.sum() == 0:
        return 0
    return (act * top).sum() / top.sum()


def _rate_top05(act, pred, weight=None):
    if len(pred.shape) == 1:
        return rate_top05_1d(act, pred, weight)
    else:
        out = []
        for col in range(pred.shape[1]):
            p = pred[:, col]
            out.append(rate_top05_1d(act, p, weight))
        return max(out)


@metricify(RATE_TOP05_PCT, "Response rate in top 5% highest predictions",
           DIRECTION_DESCENDING, match_func=match_binary)
def rate_top05(act, pred, weight=None):
    return _rate_top05(act, pred, weight)


@metricify(RATE_TOP05_PCT_W, "Response rate in top 5% highest predictions",
           DIRECTION_DESCENDING, match_func=match_binary)
def rate_top05_w(act, pred, weight=None):
    return _rate_top05(act, pred, weight)


# --------------------------------------------------------------------
def rate_toptenth_1d(act, pred, weight=None):
    """
        response rate in top 0.1% highest predictions
    """
    if len(pred.shape) > 1:
        pred = pred.ravel()
    top = (pred >= np.percentile(pred, 99.9))
    if top.sum() == 0:
        return 0
    return (act * top).sum() / top.sum()


@metricify(RATE_TOPTENTH_PCT, "Response rate in top 0.1% highest predictions",
           DIRECTION_DESCENDING, match_func=match_disabled)
def rate_toptenth(act, pred, weight=None):
    if len(pred.shape) == 1:
        return rate_toptenth_1d(act, pred)
    else:
        out = []
        for col in range(pred.shape[1]):
            p = pred[:, col]
            out.append(rate_toptenth_1d(act, p))
        return max(out)


# --------------------------------------------------------------------
def _AMS(act, pred, pct_threshold, weight=None):
    """ AMS metric - for Higgs  """
    if len(pred.shape) > 1:
        pred = pred.ravel()
    # Regularization term
    bReg = 10.0
    n = len(pred)
    ranks = n - np.argsort(pred).argsort()
    thresh = n * pct_threshold
    s = ranks <= thresh
    b = np.invert(s)
    # We need to weight by the number of rows
    # in the training set for Kaggle HIGGS
    weight_factor = float(250000. / n)
    # weight_factor = 1.
    if weight is not None:
        s_score = np.sum(weight[s & (act == 1)]) * weight_factor
        b_score = np.sum(weight[b & (act == 0)]) * weight_factor
    else:
        s_score = np.sum(np.ones((n))[s & (act == 1)]) * weight_factor
        b_score = np.sum(np.ones((n))[b & (act == 0)]) * weight_factor
    return np.sqrt(2 * ((s_score + b_score + bReg) *
                        np.log(1 + s_score / (b_score + bReg)) - s_score))


# @metricify(AMS_15,
#           "Measures the Median of Estimated Signficance with a 15% Threshold",
#           DIRECTION_DESCENDING, match_func=match_logloss)
def ams15(act, pred, weight=None):
    return _AMS(act, pred, 0.15, weight)


# @metricify(AMS_OPT,
#           "Measures the Median of Estimated Signficance with a Optimal Threshold",
#           DIRECTION_DESCENDING, match_func=match_logloss)
def amsopt(act, pred, weight=None):
    curr_best_score = 0
    for tsh in range(0, 101):
        score = _AMS(act, pred, tsh / 100., weight)
        if score > curr_best_score:
            curr_best_score = score
    return curr_best_score


# ----------------------------------------------------------------------------------------------
#
#     Regression Metrics
#
# ----------------------------------------------------------------------------------------------
def _r_squared(act, pred, weight=None):
    """
        r2 = (1-u/v), u = sum((act - pred)**2), v = sum( (act - mean(act))**2 )
    """
    if len(pred.shape) > 1:
        pred = pred.ravel()
    d = act - pred
    a_m = act - w_mean(act, weight)
    if weight is not None:
        u = np.dot(d, d * weight)
        v = np.dot(a_m, a_m * weight)
    else:
        u = np.dot(d, d)
        v = np.dot(a_m, a_m)
    if v > 0:
        return 1 - u / v
    else:
        return 0


@metricify(R_SQUARED, "Measures the proportion of total variation "
                      "of outcomes explained by the model",
           DIRECTION_DESCENDING, match_func=match_regression,
           weighted_version=R_SQUARED_W)
def r_squared(act, pred, weight=None):
    return _r_squared(act, pred, weight)


@metricify(R_SQUARED_W, "Measures the proportion of total variation "
                        "of outcomes explained by the model",
           DIRECTION_DESCENDING, match_func=match_regression_w)
def r_squared_w(act, pred, weight=None):
    return _r_squared(act, pred, weight)


# --------------------------------------------------------------------
def _rmse(act, pred, weight=None):
    """
        RMSE = Root Mean Squared Error = sqrt( mean( (act - pred)**2 ) )
    """
    if len(pred.shape) > 1:
        if pred.shape[1] == 2:
            pred = pred[:, 1]
        else:
            pred = pred.ravel()

    pred = pred.astype(np.float64, copy=False)
    d = act - pred
    # if weight is not None:
    #    d = d*weight
    # mse = np.power(d,2).mean()
    sd = np.power(d, 2)
    if weight is not None:
        if weight.sum() == 0:
            return 0
        sd = sd * weight / weight.mean()
    mse = sd.mean()
    rmse = np.sqrt(mse)

    # [MMSQUAD-2453] in case of data type overflow, check RMSE value for non and return max float64
    if np.isnan(rmse):
        return np.finfo(np.float64).max
    else:
        return rmse


@metricify(RMSE, "Measures Inaccuracy of predicted mean values",
           DIRECTION_ASCENDING, match_func=match_any, weighted_version=RMSE_W)
def rmse(act, pred, weight=None):
    return _rmse(act, pred, weight)


@metricify(RMSE_W, "Measures Inaccuracy of predicted mean values",
           DIRECTION_ASCENDING, match_func=match_any_w)
def rmse_w(act, pred, weight=None):
    return _rmse(act, pred, weight)


# --------------------------------------------------------------------
@metricify(UNRMSE, "Measures Inaccuracy of predicted mean values - Normalized by user id",
           DIRECTION_ASCENDING, grouped=InputType.UID,
           weighted_version=UNRMSE_W)
def unrmse(act, pred, weight=None):
    return _rmse(act, pred, weight)


@metricify(UNRMSE_W, "Measures Inaccuracy of predicted mean values - Normalized by user id",
           DIRECTION_ASCENDING, grouped=InputType.UID)
def unrmse_w(act, pred, weight=None):
    return _rmse(act, pred, weight)


@metricify(CURMSE, "Measures Inaccuracy of predicted mean values - only cold-start users",
           DIRECTION_ASCENDING, coldstart=InputType.UID)
def curmse(act, pred, weight=None):
    if weight is None:
        raise ValueError('curmse expects weights for cold-users')
    return _rmse(act, pred, weight)


# --------------------------------------------------------------------
def _mad(act, pred, weight=None):
    """
        MAD = Mean Absolute Deviation = mean( abs(act-pred) )
    """
    if len(pred.shape) > 1:
        pred = pred.ravel()
    d = act - pred
    if weight is not None:
        if weight.sum() == 0:
            return 0
        d = d * weight / weight.mean()
    return np.absolute(d).mean()


# We use MAE instead of MAD going forward
# MAD should not match anything

@metricify(MAD,
           "Mean Absolute Deviation (Measures Inaccuracy of predicted median values)",
           DIRECTION_ASCENDING, match_func=match_disabled,
           weighted_version=MAD_W)
def mad(act, pred, weight=None):
    return _mad(act, pred, weight)


@metricify(MAD_W,
           "Mean Absolute Deviation (Measures Inaccuracy of predicted median values)",
           DIRECTION_ASCENDING, match_func=match_disabled)
def mad_w(act, pred, weight=None):
    return _mad(act, pred, weight)


@metricify(MAE,
           "Mean Absolute Error (Measures Inaccuracy of predicted median values)",
           DIRECTION_ASCENDING, match_func=match_regression,
           weighted_version=MAE_W)
def mae(act, pred, weight=None):
    return _mad(act, pred, weight)


@metricify(MAE_W,
           "Mean Absolute Error (Measures Inaccuracy of predicted median values)",
           DIRECTION_ASCENDING, match_func=match_regression_w)
def mae_w(act, pred, weight=None):
    return _mad(act, pred, weight)


# --------------------------------------------------------------------
@metricify(UNMAD,
           "Mean Absolute Deviation (Measures Inaccuracy of predicted median values)"
           " - Normalized by user id",
           DIRECTION_ASCENDING, grouped=InputType.UID,
           weighted_version=UNMAD_W, match_func=match_disabled)
def unmad(act, pred, weight=None):
    return _mad(act, pred, weight)


@metricify(UNMAD_W,
           "Mean Absolute Deviation (Measures Inaccuracy of predicted median values)"
           " - Normalized by user id",
           DIRECTION_ASCENDING, grouped=InputType.UID, match_func=match_disabled)
def unmad_w(act, pred, weight=None):
    return _mad(act, pred, weight)


@metricify(CUMAD,
           "Mean Absolute Deviation (Measures Inaccuracy of predicted median values)"
           " - only cold-start users",
           DIRECTION_ASCENDING, coldstart=InputType.UID, match_func=match_disabled)
def cumad(act, pred, weight=None):
    if weight is None:
        raise ValueError('cumad expects weights for cold-users')
    return _mad(act, pred, weight)


@metricify(UNMAE,
           "Mean Absolute Error (Measures Inaccuracy of predicted median values)"
           " - Normalized by user id",
           DIRECTION_ASCENDING, grouped=InputType.UID,
           weighted_version=UNMAE_W)
def unmae(act, pred, weight=None):
    return _mad(act, pred, weight)


@metricify(UNMAE_W,
           "Mean Absolute Error (Measures Inaccuracy of predicted median values)"
           " - Normalized by user id",
           DIRECTION_ASCENDING, grouped=InputType.UID)
def unmae_w(act, pred, weight=None):
    return _mad(act, pred, weight)


@metricify(CUMAE,
           "Mean Absolute Error (Measures Inaccuracy of predicted median values)"
           " - only cold-start users",
           DIRECTION_ASCENDING, coldstart=InputType.UID)
def cumae(act, pred, weight=None):
    if weight is None:
        raise ValueError('cumae expects weights for cold-users')
    return _mad(act, pred, weight)


# --------------------------------------------------------------------
def _rmsle(act, pred, weight=None):
    """
        RMSLE = Root Mean Squared Logarithmic Error
            = sqrt( mean( ( log(pred+1)-log(act+1) )**2))

        ONLY WORKS FOR POSITIVE RESPONSES
    """
    if len(pred.shape) > 1:
        pred = pred.ravel()
    pred = np.maximum(pred, 0)  # ensure predictions are non-negative
    act = np.maximum(act, 0)  # ensure actuals are non-negative
    lp = np.log(pred + 1)
    la = np.log(act + 1)
    d = lp - la

    sd = np.power(d, 2)
    if weight is not None:
        if weight.sum() == 0:
            return 0
        sd = sd * weight / weight.mean()
    msle = sd.mean()
    return np.sqrt(msle)


@metricify(RMSLE, "Measures Inaccuracy of predicted median values "
                  "when the target is skewed and lognormally distributed",
           DIRECTION_ASCENDING, match_func=match_rmsle,
           weighted_version=RMSLE_W)
def rmsle(act, pred, weight=None):
    return _rmsle(act, pred, weight)


@metricify(RMSLE_W, "Measures Inaccuracy of predicted median values "
                    "when the target is skewed and lognormally distributed",
           DIRECTION_ASCENDING, match_func=match_rmsle_w)
def rmsle_w(act, pred, weight=None):
    return _rmsle(act, pred, weight)


# --------------------------------------------------------------------
def _poisson_deviance(act, pred, weight=None):
    """
        Poisson Deviance = 2*(act*log(act/pred)-(act-pred))

        ONLY WORKS FOR POSITIVE RESPONSES
    """
    if len(pred.shape) > 1:
        pred = pred.ravel()
    pred = np.maximum(pred, 1e-8)  # ensure predictions are strictly positive
    act = np.maximum(act, 0)  # ensure actuals are non-negative
    d = np.zeros(len(act))
    d[act == 0] = pred[act == 0]
    cond = act > 0
    d[cond] = act[cond] * np.log(act[cond] / pred[cond]) - (act[cond] - pred[cond])
    d = d * 2
    if weight is not None:
        if weight.sum() == 0:
            return 0
        d = d * weight / weight.mean()
    return d.mean()


@metricify(POISSON_DEVIANCE, "Measures Inaccuracy of "
                             "predicted mean values when the target is skewed to the right",
           DIRECTION_ASCENDING, match_func=match_poisson,
           weighted_version=POISSON_DEVIANCE_W)
def poisson_deviance(act, pred, weight=None):
    return _poisson_deviance(act, pred, weight)


@metricify(POISSON_DEVIANCE_W, "Measures Inaccuracy of "
                               "predicted mean values when the target is skewed to the right",
           DIRECTION_ASCENDING, match_func=match_poisson_w)
def poisson_deviance_w(act, pred, weight=None):
    return _poisson_deviance(act, pred, weight)


# --------------------------------------------------------------------
def _fve_poisson(act, pred, weight=None):
    v = _poisson_deviance(act, w_mean(act, weight), weight)
    if v > 0:
        return (1 - _poisson_deviance(act, pred, weight) / v)
    else:
        return 0


@metricify(FVE_POISSON, "Fraction of variance explained for Poisson deviance",
           DIRECTION_DESCENDING, match_func=match_poisson,
           weighted_version=FVE_POISSON_W)
def fve_poisson(act, pred, weight=None):
    return _fve_poisson(act, pred, weight)


@metricify(FVE_POISSON_W, "Fraction of variance explained for Poisson deviance",
           DIRECTION_DESCENDING, match_func=match_poisson_w)
def fve_poisson_w(act, pred, weight=None):
    return _fve_poisson(act, pred, weight)


# --------------------------------------------------------------------
def _tweedie_deviance(act, pred, weight=None, p=1.5):
    """
        ONLY WORKS FOR POSITIVE RESPONSES
    """
    if p < 1 or p > 2:
        raise ValueError('p equal to %s is not supported' % p)

    if len(pred.shape) > 1:
        pred = pred.ravel()

    if p == 1:
        return poisson_deviance(act, pred, weight)
    if p == 2:
        return gamma_deviance(act, pred, weight)
    if p == 0:
        d = (act - pred) ^ 2.0
        return d.mean()
    pred = np.maximum(pred, 1e-8)  # ensure predictions are strictly positive
    act = np.maximum(act, 0)  # ensure actuals are not negative
    d = ((act ** (2.0 - p)) / ((1 - p) * (2 - p)) -
         (act * (pred ** (1 - p))) / (1 - p) + (pred ** (2 - p)) / (2 - p))
    d = 2 * d
    if weight is not None:
        if weight.sum() == 0:
            return 0
        d = d * weight / weight.mean()
    deviance = d.mean()
    return deviance


@metricify(TWEEDIE_DEVIANCE, "Measures Inaccuracy of "
                             "predicted mean values when the target is zero-inflated and skewed",
           DIRECTION_ASCENDING, match_func=match_tweedie,
           weighted_version=TWEEDIE_DEVIANCE_W)
def tweedie_deviance(act, pred, weight=None, p=1.5):
    return _tweedie_deviance(act, pred, weight, p)


@metricify(TWEEDIE_DEVIANCE_W, "Measures Inaccuracy of "
                               "predicted mean values when the target is zero-inflated and skewed",
           DIRECTION_ASCENDING, match_func=match_tweedie_w)
def tweedie_deviance_w(act, pred, weight=None, p=1.5):
    return _tweedie_deviance(act, pred, weight, p)


# --------------------------------------------------------------------
def _fve_tweedie(act, pred, weight=None, p=1.5):
    v = _tweedie_deviance(act, w_mean(act, weight), weight, p)
    if v > 0:
        return (1 - _tweedie_deviance(act, pred, weight, p) / v)
    else:
        return 0


@metricify(FVE_TWEEDIE, "Fraction of variance explained for Tweedie deviance",
           DIRECTION_DESCENDING, match_func=match_tweedie,
           weighted_version=FVE_POISSON_W)
def fve_tweedie(act, pred, weight=None, p=1.5):
    return _fve_tweedie(act, pred, weight, p)


@metricify(FVE_TWEEDIE_W, "Fraction of variance explained for Tweedie deviance",
           DIRECTION_DESCENDING, match_func=match_tweedie_w)
def fve_tweedie_w(act, pred, weight=None, p=1.5):
    return _fve_tweedie(act, pred, weight, p)


# --------------------------------------------------------------------
def _gamma_deviance(act, pred, weight=None):
    """
        Gamma Deviance = 2*(-log(act/pred)+(act-pred)/pred)

        ONLY WORKS FOR POSITIVE RESPONSES
    """
    if len(pred.shape) > 1:
        pred = pred.ravel()
    pred = np.maximum(pred, 0.01)  # ensure predictions are stricly positive
    act = np.maximum(act, 0.01)  # ensure actuals are strictly positive
    d = 2 * (-np.log(act / pred) + (act - pred) / pred)
    if weight is not None:
        if weight.sum() == 0:
            return 0
        d = d * weight / weight.mean()
    return d.mean()


@metricify(GAMMA_DEVIANCE, "Measures Inaccuracy of "
                           "predicted mean values when the target is strictly "
                           "positive and highly skewed",
           DIRECTION_ASCENDING, match_func=match_gamma,
           weighted_version=GAMMA_DEVIANCE_W)
def gamma_deviance(act, pred, weight=None):
    return _gamma_deviance(act, pred, weight)


@metricify(GAMMA_DEVIANCE_W, "Measures Inaccuracy of "
                             "predicted mean values when the target is strictly "
                             "positive and highly skewed",
           DIRECTION_ASCENDING, match_func=match_gamma_w)
def gamma_deviance_w(act, pred, weight=None):
    return _gamma_deviance(act, pred, weight)


# --------------------------------------------------------------------
def _fve_gamma(act, pred, weight=None):
    v = _gamma_deviance(act, w_mean(act, weight), weight)
    if v > 0:
        return (1 - _gamma_deviance(act, pred, weight) / v)
    else:
        return 0


@metricify(FVE_GAMMA, "Fraction of variance explained for Gamma deviance",
           DIRECTION_DESCENDING, match_func=match_gamma,
           weighted_version=FVE_POISSON_W)
def fve_gamma(act, pred, weight=None):
    return _fve_gamma(act, pred, weight)


@metricify(FVE_GAMMA_W, "Fraction of variance explained for Gamma deviance",
           DIRECTION_DESCENDING, match_func=match_gamma_w)
def fve_gamma_w(act, pred, weight=None):
    return _fve_gamma(act, pred, weight)


# --------------------------------------------------------------------
def _mape(act, pred, weight=None):
    if len(pred.shape) > 1:
        pred = pred.ravel()
    # These lines to prevent errors when running on zeros.
    # Such a case should not make it to the front, but as written
    # we currently run all regression metrics on all regression problems,
    # but just hide some of them from the frontend
    actual = np.maximum(act, 0.0001)
    predic = np.maximum(pred, 0.0001)

    rel_error = (np.abs(actual - predic) / actual)
    if weight is not None:
        total_weight = weight.sum()
        rel_error = rel_error * weight
        divisor = total_weight
    else:
        divisor = len(act)
    return rel_error.sum() * 100.0 / divisor


@metricify(MAPE, 'Measures mean absolute percentage error',
           DIRECTION_ASCENDING, match_func=match_strictly_positive,
           weighted_version=MAPE_W)
def mape(act, pred, weight=None):
    return _mape(act, pred, weight)


@metricify(MAPE_W, 'Measures mean absolute percentage error',
           DIRECTION_ASCENDING, match_func=match_strictly_positive_w)
def mape_w(act, pred, weight=None):
    return _mape(act, pred, weight)


# --------------------------------------------------------------------
def _dcg(relevances, rank=10):
    """Discounted cumulative gain at rank (DCG)"""
    relevances = np.asarray(relevances)[:rank]
    n_relevances = len(relevances)
    if n_relevances == 0:
        return 0.

    discounts = np.log2(np.arange(n_relevances) + 2)
    return np.sum((2.0 ** relevances - 1) / discounts)


def _ndcg(relevances, pred, rank=10):
    """Normalized discounted cumulative gain (NDGC)"""
    relevances = np.asarray(relevances)
    n_grades = np.unique(relevances).shape[0]  # pylint: disable=no-member
    if n_grades == 1:
        # only one relevance grade -- we cannot rank it.
        #  assume worst score
        return 0.0
    # sort pred in desc order (largest predicted first)
    sort_order = pred.argsort()[::-1]
    relevances = relevances[sort_order]
    best_dcg = _dcg(sorted(relevances, reverse=True), rank)
    if best_dcg == 0:
        return 0.0

    return _dcg(relevances, rank) / best_dcg


@metricify(NDCG, "Measures ability to rank graded items",
           DIRECTION_DESCENDING, ranking=True, grouped=InputType.UID,
           skip_single_groups=True)
def ndcg(act, pred, weight=None, rank=10):
    """Normalized discounted cumulative gain. """
    return _ndcg(act, pred, rank)


# -metric for the mets problem
@metricify(R_SQUARED_20_80, "R-squared 20/80",
           DIRECTION_DESCENDING, match_func=match_disabled)
def r_squared_mets(act, pred, weight=None):
    """
        r2 = (1-u/v), u = sum((act - pred)**2), v = sum( (act - mean(act))**2 )
    """
    if len(pred.shape) > 1:
        pred = pred.ravel()
    pred = np.minimum(np.maximum(pred, 20), 80)
    d = act - pred
    a_m = act - w_mean(act, weight)
    if weight is not None:
        d = d * weight
        a_m = a_m * weight
    u = np.dot(d, d)
    v = np.dot(a_m, a_m)
    if v > 0:
        return 1 - u / v
    else:
        return 0


def direction_by_name(metric_name):
    """Returns the direction of the metric, 1 means lower is better.

    Returns
    -------
    direction : int
        Either 1 (lower is better) or -1 (higher is better)
    """
    if metric_name not in METRIC_MAP:
        raise ValueError('Metric %r is not supported' % metric_name)

    metric = METRIC_MAP[metric_name]
    return metric['direction']


def metric_by_name(metric_name):
    """Returns the metric func of the ``metric_name``.

    Returns
    -------
    func : callable(act, pred, weights)
        The metric function.
    """
    if metric_name not in METRIC_MAP:
        raise ValueError('Metric %r is not supported' % metric_name)

    metric = METRIC_MAP[metric_name]
    return metric['func']


def group_key_by_name(metric_name):
    """Returns the group key of the ``metric_name``.

    Returns
    -------
    group_key : InputType or None
        The group key of the metric or None if no
    """
    if metric_name not in METRIC_MAP:
        raise ValueError('Metric %r is not supported' % metric_name)

    metric = METRIC_MAP[metric_name]
    return metric['grouped']


def coldstart_key_by_name(metric_name):
    """Returns the coldstart key of the ``metric_name``.

    Similar to group key, a coldstart key is an InputType
    that is used to determine new groups.

    Returns
    -------
    coldstart_key : InputType or None
        The coldstart key of the metric or None if no
    """
    if metric_name not in METRIC_MAP:
        raise ValueError('Metric %r is not supported' % metric_name)

    metric = METRIC_MAP[metric_name]
    return metric['coldstart']


def coldstart_weights(coldstart_column, test):
    """Compute a binary weight vector indicating whether i is a coldstart or not.

    Parameters
    ----------
    coldstart_column : pd.DataFrame, shape=(n_samples)
        An dataframe holding the group identifiers; coldstarts are groups that only
        appear in ``test``.
    test : array-like, dtype=int
        Index array indicating the test samples

    Returns
    -------
    weight : np.ndarray, shape=(n_samples,), dtype=float
        Either 1.0 if the row belongs to a coldstart or 0 otherwise.
    """
    # if coldstart_column.shape[1] != 1:
    #    raise ValueError('coldstat_column has %d columns but expected 1' %
    #                     coldstart_column.shape[1])
    # Now, we are using this such that this is always passed a Series
    train = np.ones(coldstart_column.shape[0], dtype=np.bool)
    train[test] = False
    train = np.where(train)[0].tolist()

    test_groups = set(coldstart_column.iloc[test])

    # if train group is empty (e.g. 100% samplepct run, (-1,-1)) then coldstart == no-coldstart
    if len(train) > 0:
        train_groups = set(coldstart_column.iloc[train])
    else:
        train_groups = set()

    coldstart_groups = test_groups.difference(train_groups)

    weight = coldstart_column.isin(coldstart_groups).values.astype(np.float)

    logger.debug('Coldstart n_coldstarts:%d n_users:%d n_samples:%d ratio:%f',
                 len(coldstart_groups),
                 len(train_groups), weight.sum(),
                 len(test) / (1.0 + len(train)))
    return weight


metric_normalization = {
    ACCURACY: 'zero_to_one_descending',
    ACCURACY_W: 'zero_to_one_descending',
    BALANCED_ACCURACY: 'zero_to_one_descending',
    BALANCED_ACCURACY_W: 'zero_to_one_descending',
    'AUC': 'point_five_to_one_descending',
    'Weighted AUC': 'point_five_to_one_descending',
    'Kolmogorov-Smirnov': 'zero_to_one_descending',
    'Weighted Kolmogorov-Smirnov': 'zero_to_one_descending',
    'Weighted Gini Norm': 'zero_to_one_descending',
    'Weighted R Squared': 'zero_to_one_descending',
    'Gini Norm': 'zero_to_one_descending',
    'Weighted Normalized Gini': 'zero_to_one_descending',
    'R Squared': 'zero_to_one_descending',
    'R Squared 20/80': 'zero_to_one_descending',
    'FVE Binomial': 'zero_to_one_descending',
    'FVE Poisson': 'zero_to_one_descending',
    'FVE Gamma': 'zero_to_one_descending',
    'FVE Tweedie': 'zero_to_one_descending',
    'Weighted FVE Binomial': 'zero_to_one_descending',
    'Weighted FVE Poisson': 'zero_to_one_descending',
    'Weighted FVE Gamma': 'zero_to_one_descending',
    'Weighted FVE Tweedie': 'zero_to_one_descending',
    'Rate@Top10%': 'zero_to_one_descending_non_zero_null',
    'Rate@Top5%': 'zero_to_one_descending_non_zero_null',
    'Weighted Rate@Top10%': 'zero_to_one_descending_non_zero_null',
    'Weighted Rate@Top5%': 'zero_to_one_descending_non_zero_null',
    'Rate@TopTenth%': 'zero_to_one_descending',
    'MAPE': 'zero_to_one_ascending_non_zero_null',
    'NDCG': 'zero_to_one_descending',
    'Ians Metric': 'zero_to_one_descending',
    'Weighted Normalized RMSE': 'zero_to_one_descending',
    'Weighted Normalized MAD': 'zero_to_one_descending',
    'Weighted MAPE': 'zero_to_one_ascending_non_zero_null',
    'Weighted Gini': 'open_descending',
    'AMS@15%tsh': 'open_descending',
    'AMS@opt_tsh': 'open_descending',
    'Gini': 'open_descending',
    'LogLoss': 'open_ascending',
    'LogLossSig': 'open_ascending',
    'MAD': 'open_ascending',
    'MAE': 'open_ascending',
    'RMSE': 'open_ascending',
    'RMSLE': 'open_ascending',
    'Poisson Deviance': 'open_ascending',
    'Tweedie Deviance': 'open_ascending',
    'Gamma Deviance': 'open_ascending',
    'Normalized RMSE': 'open_ascending',
    'Normalized MAD': 'open_ascending',
    'Coldstart RMSE': 'open_ascending',
    'Coldstart MAD': 'open_ascending',
    'Weighted LogLoss': 'open_ascending',
    'Weighted LogLossSig': 'open_ascending',
    'Weighted RMSE': 'open_ascending',
    'Weighted MAD': 'open_ascending',
    'Weighted MAE': 'open_ascending',
    'Weighted RMSLE': 'open_ascending',
    'Weighted Poisson Deviance': 'open_ascending',
    'Weighted Tweedie Deviance': 'open_ascending',
    'Weighted Gamma Deviance': 'open_ascending',
}


def normalize_ace_score(metric_name, var_score, null_score):
    """
        make scores from ACE range from 0 (no info) to 1 (perfect info).
        scores may be slightly less than zero in the case of overfitting
        low info variables.
    """
    if metric_name not in metric_normalization:
        raise ValueError('Metric not supported')
    if metric_normalization[metric_name] == 'point_five_to_one_descending':
        # metrics ranging from 0.5 to 1, higher is better
        return 2 * (var_score - 0.5)
    if metric_normalization[metric_name] == 'zero_to_one_descending':
        # metrics ranging from 0 to 1, higher is better, null model ~= 0
        return var_score
    if metric_normalization[metric_name] == 'zero_to_one_ascending_non_zero_null':
        # metrics ranging from 0 to 1, lower is better & null score >= 0
        return 2 / (1 + exp(-0.1 * (null_score / max(0.001, var_score) - 1))) - 1
    if metric_normalization[metric_name] == 'zero_to_one_descending_non_zero_null':
        # metrics ranging from 0 to 1, higher is better & null score >= 0
        return 2 / (1 + exp(-0.1 * (var_score / max(0.001, null_score) - 1))) - 1
    if metric_normalization[metric_name] == 'open_descending':
        # metrics ranging from -inf to +inf, higher is better
        return 1 - null_score / var_score
    if metric_normalization[metric_name] == 'open_ascending':
        # metrics ranging from -inf to +inf, lower is better
        return 1 - var_score / null_score
    raise ValueError("Normalization type not supported")


# ----------------------------------------------------------------------------------------------
#
#     Metrics Report
#
# ----------------------------------------------------------------------------------------------

@accepts(None, None, Container, np.ndarray)
def metrics_report(parts, metrics_keys, pred, y, Z, weight=None):
    """
        This function calculates all requested metrics (input = mets),
        on all possible test partitions available (input = modelrep)

        Inputs:
        parts = the trained partitions
        metric_map = a dictionary listing the metrics to calculate, for example:
            mets = {'logloss':logloss_function, 'AUC':auc_function}
        pred = model predictions
        y = response
        Z = partition object
        build_data : BuildData object or None
            Used to extract the groups for grouped metrics


        Output:
        Return a dictionary with the following structure:
            out['labels'] = list of test partition labels (length = N)
            for each 'metric name':
                out['metric name'] = list of metric values by partition label (length = N)
    """
    metrics_map = {key: metric_by_name(key) for key in metrics_keys}

    calc = dict((mk, {}) for mk in
                metrics_keys)  # temp storage of calculated metrics by partition
    # calc[ metric_key ][ partition_key ] = calculated value

    rounding = {'MAD': 5, 'RMSE': 5}  # default = 5

    def _compute_metrics(y, p, Z, r=None, k=None, w=None, out=None, key=None,
                         roundit=True,
                         test_rows=None):
        """Convencience function to compute metrics for y and p.

        Computes self.metrics on test rows of response y and predictions p

        Parameters
        ----------
        y : np.array, shape=(n,)
            Target response
        p : np.array, shape=(n,)
            The predictions
        Z : Partition
            The partition object
        r : int
            The repetition of Z
        k : int
            The fold of Z
        w : np.ndarray or None, shape=(n,)
            Weights (optional)
        out : dict or None
            A dict where the output is stored under out[metric].append(score)
        key : dict-key or None
            An optional key that is used to store the metric: out[metric][key] = score
        roundit : bool
            Whether or not to round
        test_rows : array-like, dtype=int
            Alternative to Z, r, k -- if r==None and k==None
        """
        # FIXME this might not be the case
        #  see tests/IntegrationTests/test_secure_worker_unittest.py:test_model_blending
        # assert y.shape[0] == p.shape[0]

        if w is not None:
            assert y.shape[0] == w.shape[0]
            if isinstance(w, pd.Series):
                w = w.values
        if out is None:
            out = defaultdict(list)

        if r is None and k is None and test_rows is None:
            raise ValueError(
                'Either r and k must not be None or test_rows must be specified')

        if test_rows is None:
            test_rows = Z.S(r=r, k=k)

        y_ = y[test_rows]
        p_ = p[test_rows]

        for mk in metrics_keys:
            if w is not None:
                w_ = w.copy()
            else:
                w_ = None
            metric_func = metrics_map[mk]

            group_key = group_key_by_name(mk)
            groups = None
            if group_key:
                # we piggy-back dataset_id with the object
                groups = pred.get(group_key)
                assert groups is not None
                assert groups.shape[0] == y.shape[0], '%d != %d' % (groups.shape[0], y.shape[0])
                groups = groups.values[test_rows]

            coldstart_key = coldstart_key_by_name(mk)
            if coldstart_key:
                coldstart_col = pred.get(coldstart_key)
                assert coldstart_col is not None
                assert coldstart_col.shape[0] == y.shape[0], \
                    '%d != %d' % (coldstart_col.shape[0], y.shape[0])

                if w_ is None:
                    w_ = coldstart_weights(coldstart_col, test_rows)
                else:
                    w_ *= coldstart_weights(coldstart_col, test_rows)

            if w_ is None:
                score = metric_func(y_, p_, groups=groups)
            else:
                w_ = w_[test_rows]
                score = metric_func(y_, p_, w_, groups=groups)

            if roundit:
                score = round(score, rounding.get(mk, 5))
            if key:
                out[mk][key] = score
            else:
                out[mk].append(score)

        return out

    # calculate metrics
    parts_keys = []
    for p in parts:
        key = (p['r'], p['k'])  # partition key
        parts_keys.append(key)
        rows = Z.S(
            **p)  # row mask to select test partition to calculate leaderboard metrics
        if p['r'] > -1:
            _compute_metrics(y, pred(**p), Z, p['r'], p['k'], weight, calc,
                             key)

    out = {}

    out['labels'] = []
    out['metrics'] = metrics_keys
    for mk in metrics_keys:  # mk is the metric label (str)
        out[
            mk] = []  # list of metric values, one for each partition listed in labels

    # if (0,0) is build:
    #  FIXME I think this never happens (Peter)
    if (0, 0) in parts_keys:
        assert False
        out['labels'].append('(0,0)')
        for mk in metrics_keys:
            out[mk].append(calc[mk][(0, 0)])

    # if rep1 is fully built:
    if len([i for i in parts_keys if
            i[0] == 0]) == Z.max_reps and Z.n_reps != 1:
        assert False
    #     out['labels'].append( '(0,.)' )
    #     for mk in metrics_keys:
    #         tmp = calc[mk]
    #         rn = rounding.get(mk,5)
    #         out[mk].append( round( np.mean( [tmp[k] for k in parts_keys if k[0]==0 ] ), rn) )

    #     #avg pred:
    #     n = 1.0
    #     for p in [j for j in parts if (j['r']==0)*(j['k']!=-1)]:
    #         if n==1: p_avg = np.copy(pred(**p))
    #         else: p_avg += pred(**p)
    #         n += 1
    #         rows = Z.S(**p)
    #     p_avg = p_avg/n

    #     out['labels'].append( '(0,A)' )
    #     _compute_metrics(y, p_avg, Z, p['r'], p['k'], weight, out, key=None)

    # if rep1 has k=-1 built:
    if (0, -1) in parts_keys or (-1, -1) in parts_keys:
        r = -1 if (-1, -1) in parts_keys else 0
        out['labels'].append('(0,-1)')
        _compute_metrics(y, pred(r=r, k=-1), Z, 0, -1, weight, out, key=None)

    # Commenting out since not used AFAIK (Peter)
    # if all reps are built:
    if len([i for i in parts_keys if (i[0] != -1) * (i[1] != -1)]) == \
            Z.max_reps * Z.max_folds and Z.max_reps * Z.max_folds > 0:
        assert False
    #     out['labels'].append( '(.,.)' )
    #     for mk in metrics_keys:
    #         tmp = calc[mk]
    #         rn = rounding.get(mk,5)
    #         out[mk].append( round( np.mean( [tmp[k] for k in parts_keys
    #                                          if (k[0]!=-1)*(k[1]!=-1)] ), rn) )
    #     #avg pred:
    #     A = dict( (mk, []) for mk in metrics_keys )
    #     for rep in [i for i in parts.reps if i!=-1]:
    #         n = 1.0
    #         for p in [j for j in parts if (j['r']==rep)*(j['k']!=-1)]:
    #             if n==1: p_avg = np.copy(pred(**p))
    #             else: p_avg += pred(**p)
    #             n += 1
    #             rows = Z.S(**p)
    #         p_avg = p_avg/n
    #         _compute_metrics(y, p_avg, Z, p['r'], p['k'], weight, out=A, key=None, roundit=False)

    #     out['labels'].append( '(.,A)' )
    #     for mk in metrics_keys:
    #         rn = rounding.get(mk, 5)
    #         out[mk].append(round(np.mean(A[mk]), rn))

    # The 5-Fold CV case -- we average the scores of the 5 Folds
    req = set([(r, -1) for r in range(Z.max_reps)])
    if req & set(parts_keys) == req and len(req) > 1:
        tmp = defaultdict(list)
        # compute metric score for each fold
        for p in [i for i in parts if (i['k'] == -1) * (i['r'] != -1)]:
            _compute_metrics(y, pred(**p), Z, p['r'], p['k'], weight, out=tmp,
                             roundit=False)

        out['labels'].append('(.,-1)')
        # store mean 5-Fold CV in out
        for mk, scores in tmp.iteritems():
            out[mk].append(round(np.mean(scores), rounding.get(mk, 5)))

            # # FIXME old way was to concat the predictions / targets and then compute the metric
            # # Different results for Gini Norm etc.
            # row_stack = []
            # pred_ = np.empty_like(y)
            # pred_.fill(np.nan)
            # for p in [i for i in parts if (i['k']==-1)*(i['r']!=-1)]:
            #     rows = Z.S(**p)
            #     pred_[rows] = pred(**p)[rows]
            #     row_stack.extend(rows)

            # assert len(row_stack) == y.shape[0]
            # assert len(set(row_stack)) == y.shape[0]
            # out['labels'].append( '(.,-1)' )
            # _compute_metrics(y, pred_, Z, None, None, weight, out=out, key=None,
            #                  test_rows=row_stack)

    if (-1, -1) in parts_keys:
        row_stack = []
        pred_ = np.empty_like(y)
        pred_.fill(np.nan)
        for p in range(Z.max_reps):
            rows = Z.S(r=p, k=-1)
            pred_[rows] = pred(r=-1, k=-1)[rows]
            row_stack.extend(rows)
        out['labels'].append('(.,-1)')

        _compute_metrics(y, pred_, Z, None, None, weight, out=out, key=None,
                         test_rows=row_stack)

    # check for infinite metric scores
    for mk in metrics_keys:
        if not np.all(np.isfinite(out[mk])):
            raise ValueError('metric %r is not finite' % mk)

    return out


def serialize_metric_map_with_match_func_names():
    """Create a dictionary containing METRIC_MAP with functions converted to
    string representations and a md5 checksum of metric_match_funcs.py.
    """
    metric_cpy = copy.deepcopy(METRIC_MAP)

    for _, entry in metric_cpy.iteritems():
        # get rid of funtion
        entry.pop('func')
        # change match_func into name
        entry['match_func'] = entry['match_func'].__name__

    # get checksum
    engine_dir = "/".join(os.path.abspath(__file__).split("/")[:-1])
    match_fn_file = os.path.join(engine_dir, 'match_funcs.py')
    checksum = hashlib.md5(open(match_fn_file, 'rb').read()).hexdigest()

    result = {'metric_map': metric_cpy, 'version': checksum}
    return result


def main():
    """Write metric definitions to file

    This needs to be called whenever this file is updated."""
    outfp = open('../../common/engine/metrics.json', 'w')
    metrics = []
    for metric in METRIC_MAP:
        entry = METRIC_MAP[metric].copy()
        entry.pop('func')
        entry.pop('match_func')
        metrics.append(entry)
    json.dump(metrics, outfp)
    outfp.close()


if __name__ == '__main__':
    main()
