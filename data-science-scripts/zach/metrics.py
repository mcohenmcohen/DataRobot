import numpy as np
from sklearn.metrics import roc_auc_score as auc_score

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

def auc(act, pred, weight=None):
    return _auc(act, pred, weight)

def auc_w(act, pred, weight=None):
    return _auc(act, pred, weight)

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

_gini_norm(np.array([1, 1, 0, 0]), np.array([1,0,0,0]))
