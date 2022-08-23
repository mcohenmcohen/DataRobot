import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.outliers_influence as oi
import statsmodels.stats.api as sms
from sklearn.utils.multiclass import type_of_target
from scipy import stats
from numpy.testing import assert_almost_equal, assert_equal
from tabulate import tabulate
import math
import json


def ramsey_RESET_test(data, target, features, degree, plot_on=False):
    '''
    Perform a ramsey RESET test using statsmodels.stats.outliers_influence

    Parameters:
        data - pandas dataframe
        target - target variable name
        features - features to test
        plot - flag to plot or not

    Returns
        oi.reset_ramsey(ols_model) result, which is a statsmodels.stats.contrast.ContrastResults

    Code ref:  http://nullege.com/codes/show/src%40s%40t%40statsmodels-HEAD%40statsmodels%40examples%40ex_outliers_influence.py/95/statsmodels.stats.outliers_influence.reset_ramsey/python

    Other refs:
        https://github.com/statsmodels/statsmodels/blob/master/statsmodels/stats/outliers_influence.py
        https://programtalk.com/python-examples/statsmodels.stats.outliers_influence.reset_ramsey/
    '''
    endog = data[target]
    exog = sm.add_constant(data[features])

    res_ols = sm.OLS(endog, exog).fit()

    hh = (res_ols.model.exog * res_ols.model.pinv_wexog.T).sum(1)
    x = res_ols.model.exog
    hh_check = np.diag(np.dot(x, np.dot(res_ols.model.normalized_cov_params, x.T)))

    assert_almost_equal(hh, hh_check, decimal=13)

    res = res_ols  # alias

    # http://en.wikipedia.org/wiki/PRESS_statistic
    # predicted residuals, leave one out predicted residuals
    resid_press = res.resid / (1-hh)
    ess_press = np.dot(resid_press, resid_press)

    sigma2_est = np.sqrt(res.mse_resid)  # can be replace by different estimators of sigma
    sigma_est = np.sqrt(sigma2_est)
    resid_studentized = res.resid / sigma_est / np.sqrt(1 - hh)
    # http://en.wikipedia.org/wiki/DFFITS:
    dffits = resid_studentized * np.sqrt(hh / (1 - hh))

    nobs, k_vars = res.model.exog.shape
    # Belsley, Kuh and Welsch (1980) suggest a threshold for abs(DFFITS)
    dffits_threshold = 2 * np.sqrt(k_vars/nobs)

    res_ols.df_modelwc = res_ols.df_model + 1
    n_params = res.model.exog.shape[1]
    # http://en.wikipedia.org/wiki/Cook%27s_distance
    cooks_d = res.resid**2 / sigma2_est / res_ols.df_modelwc * hh / (1 - hh)**2
    # or
    # Eubank p.93, 94
    cooks_d2 = resid_studentized**2 / res_ols.df_modelwc * hh / (1 - hh)
    # threshold if normal, also Wikipedia
    alpha = 0.1
    # df looks wrong
    # print('scipy inverse survival function 1-alpha:', stats.f.isf(1-alpha, n_params, res.df_resid))
    # print('scipy survival function cooks_d', stats.f.sf(cooks_d, n_params, res.df_resid))
    #
    # print('Cooks Distance:')
    # print(cooks_d)
    # print(cooks_d2)

    if plot_on:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(3, 1, 2)
        plt.plot(cooks_d, 'o', label="Cook's distance")
        plt.legend(loc='upper left')
        ax2 = fig.add_subplot(3, 1, 3)
        plt.plot(resid_studentized, 'o', label='studentized_resid')
        plt.plot(dffits, 'o', label='DFFITS')
        leg = plt.legend(loc='lower left', fancybox=True)
        leg.get_frame().set_alpha(0.5)  # , fontsize='small')
        ltext = leg.get_texts()  # all the text.Text instance in the legend
        plt.setp(ltext, fontsize='small')  # the legend text fontsize

    rr_res = oi.reset_ramsey(res, degree=degree)
    # print('oi.reset_ramsey:', rr_res)

    # infl = oi.OLSInfluence(res_ols)
    # print(infl.resid_studentized_external)
    # print(infl.resid_studentized_internal)
    # print(infl.summary_table())
    # print(oi.summary_table(res, alpha=0.05)[0])

    return rr_res


def compare_ftest(contrast_res, other, decimal=(5, 4)):
    '''
    Ref:
        https://programtalk.com/vs2/python/12423/statsmodels/statsmodels/regression/tests/test_glsar_gretl.py/#

    Ex:
        compare_ftest(ols_model, oi.ramsey_RESET(ols_model, 3), (2,4))
    '''
    assert_almost_equal(contrast_res.fvalue, other.fvalue, decimal=decimal[0])
    assert_almost_equal(contrast_res.pvalue, other.pvalue, decimal=decimal[1])
    assert_equal(contrast_res.df_num, other[2])
    assert_equal(contrast_res.df_denom, other[3])
    assert_equal("f", other[4])


def chow_test(df, features, target, split_col, split_val, print_on=True):
    '''
    Perform a chow test.

    Parameters:
        df - Dataframe
        features - list of feature names
        target - String, the target variable name
        split_col - Column to use for the split, likely a date column
        split_val - Value to use for the split, likely a date

    Returns:
        F-value: Float value of chow break test
    '''
    def find_rss(data):
        X = data[features]
        y = data[target]
        X = sm.add_constant(X)
        ols_model = sm.OLS(y, X).fit()
        rss = ols_model.ssr
        return rss

    rss_total = find_rss(df)

    df1 = df[df[split_col] < split_val]
    n_1 = df1.shape[0]
    rss_1 = find_rss(df1)

    df2 = df[df[split_col] >= split_val]
    n_2 = df2.shape[0]
    rss_2 = find_rss(df2)

    chow_nom = (rss_total - (rss_1 + rss_2)) / 2
    chow_denom = (rss_1 + rss_2) / (n_1 + n_2 - 4)

    result = chow_nom / chow_denom

    if print_on:
        print('Calculation:')
        print('- chow_nom = (rss_total - (rss_1 + rss_2)) / 2')
        print('- chow_denom = (rss_1 + rss_2) / (n_1 + n_2 - 4)')
        print()
        print('- chow_nom = (%s - (%s + %s)) / 2' % (rss_total, rss_1, rss_2))
        print('- chow_denom = (%s + %s) / (%s + %s - 4)' % (rss_1, rss_2, n_1, n_2))
        print()
        print('= %s' % result)

    return result


def chow_test2(y1, x1, y2, x2):
    '''
    Perform a chow test.
        ref: https://github.com/jtloong/chow_test

    Parameters:
        y1 - An array-like variable representing y-value data before the proposed break point
        x1 - An array-like variable representing x-value data before the proposed break point
        y2 - An array-like variable representing y-value data after the proposed break point
        x2 - An array-like variable representing x-value data after the proposed break point

    Returns:
        F-value: Float value of chow break test
    '''
    def find_rss(y, x):
        '''This is the subfunction to find the residual sum of squares for a given set of data
        Args:
            y: Array like y-values for data subset
            x: Array like x-values for data subset

        Returns:
            rss: Returns residual sum of squares of the linear equation represented by that data
            length: The number of n terms that the data represents
        '''
        A = np.vstack([x, np.ones(len(x))]).T
        rss = np.linalg.lstsq(A, y)[1]
        length = len(y)
        return (rss, length)

    rss_total, n_total = find_rss(np.append(y1, y2), np.append(x1, x2))
    rss_1, n_1 = find_rss(y1, x1)
    rss_2, n_2 = find_rss(y2, x2)

    chow_nom = (rss_total - (rss_1 + rss_2)) / 2
    chow_denom = (rss_1 + rss_2) / (n_1 + n_2 - 4)

    return chow_nom / chow_denom


def breusch_pagan_test(resid, exog):
    '''
    Perform Breush-Paga test and print out results.

    Parameters:
        resid - ols residuals, Series or array
        exog - dataframe or matrix like structure
    '''
    print('(Calculation: sigma_i = sigma * f(alpha_0 + alpha z_i))')
    name = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f_pvalue']
    test = sms.het_breushpagan(resid, exog)

    table = [[n, v] for n, v in zip(name, test)]
    headers = ["Statistic", "Value"]
    print(tabulate(table, headers, tablefmt="simple"))

    return test


def het_white(resid, exog, retres=False):
    # http://www.statsmodels.org/dev/_modules/statsmodels/sandbox/stats/diagnostic.html
    '''
    ** This is the statmsmodels function, copied here in full except for an extra
       assertion line that may be omitted, per the author.  See below.
       -mc

    White's Lagrange Multiplier Test for Heteroscedasticity

    Parameters
    ----------
    resid : array_like
        residuals, square of it is used as endogenous variable
    exog : array_like
        possible explanatory variables for variance, squares and interaction
        terms are included in the auxilliary regression.
    resstore : instance (optional)
        a class instance that holds intermediate results. Only returned if
        store=True

    Returns
    -------
    lm : float
        lagrange multiplier statistic
    lm_pvalue :float
        p-value of lagrange multiplier test
    fvalue : float
        f-statistic of the hypothesis that the error variance does not depend
        on x. This is an alternative test variant not the original LM test.
    f_pvalue : float
        p-value for the f-statistic

    Notes
    -----
    assumes x contains constant (for counting dof)

    question: does f-statistic make sense? constant ?

    References
    ----------
    Greene section 11.4.1 5th edition p. 222
    now test statistic reproduces Greene 5th, example 11.3

    '''
    x = np.asarray(exog)
    y = np.asarray(resid)
    if x.ndim == 1:
        raise ValueError('x should have constant and at least one more variable')
    nobs, nvars0 = x.shape
    i0, i1 = np.triu_indices(nvars0)
    exog = x[:, i0]*x[:, i1]
    nobs, nvars = exog.shape
    assert nvars == nvars0*(nvars0-1)/2. + nvars0
    resols = sm.OLS(y**2, exog).fit()
    fval = resols.fvalue
    fpval = resols.f_pvalue
    lm = nobs * resols.rsquared
    # Note: degrees of freedom for LM test is nvars minus constant
    # degrees of freedom take possible reduced rank in exog into account
    # df_model checks the rank to determine df

    # extra calculation that can be removed:
    # - THIS EXTRA CALCULATION IS COMMENTED OUT!  My rank diff is always 1 and fails -mc
    # assert resols.df_model == np_matrix_rank(exog) - 1

    lmpval = stats.chi2.sf(lm, resols.df_model)

    name = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f_pvalue']
    test = [lm, lmpval, fval, fpval]

    table = [[n, v] for n, v in zip(name, test)]
    headers = ["Statistic", "Value"]
    print(tabulate(table, headers, tablefmt="simple"))

    return test


def anderson_darling_test(df):
    '''
    Perform the Anderson Darling test and print out results.

    Parameters:
        df - dataframe or matrix like structure

    Returns:
        (AD statistic, pvalue)
    '''
    # stats.anderson(df[feature_set])
    test = sms.normal_ad(df)

    table = [[f, s, p] for f, s, p in zip(df.columns, test[0], test[1])]
    headers = ["Feature", "Statistic", "pvalue"]
    print(tabulate(table, headers, tablefmt="simple"))

    return test

    # for feat in feature_set:
    #     result = stats.anderson(df[feat])
    #     print(feat)
    #     print('- statistic: %s, pvalues at critical values:' % result.statistic)
    #     print('  ' + '    '.join(map(str, result.significance_level.tolist())))
    #     print('  ' + '  '.join(map(str, result.critical_values.tolist())))


def multicollinearity_condition_number(df):
    '''
    Derive the condition number.
    ref: http://www.statsmodels.org/dev/examples/notebooks/generated/ols.html

    Parameters:
        df - dataframe

    Returns:
        Condition number
    '''
    X = sm.add_constant(df)
    norm_x = X.values
    for i, name in enumerate(X):
        if name == "const":
            continue
        norm_x[:, i] = X[name]/np.linalg.norm(X[name])
    norm_xtx = np.dot(norm_x.T, norm_x)
    eigs = np.linalg.eigvals(norm_xtx)
    condition_number = np.sqrt(eigs.max() / eigs.min())

    return condition_number


def VIF_test(data, target, features):
    '''
    Perform Variance Inflation Factor (VIF) test

    Parameters:
        data - pandas dataframe
        target - target variable name
        features - features to test
        plot - flag to plot or not

    Returns:
        None.  Just prints the output
    '''
    endog = data[target]
    X = data[features]
    exog = sm.add_constant(X)
    res_ols = sm.OLS(endog, exog).fit()

    res_name = []
    res_vif = []
    res_vif_dropout = []

    for i in range(len(res_ols.model.exog_names)):
        name = res_ols.model.exog_names[i]
        if name == "const":
            continue
        val = oi.variance_inflation_factor(res_ols.model.exog, i)

        res_name.append(name)
        res_vif.append(val)

        X_dropout = X.loc[:, X.columns != name]
        exog = sm.add_constant(X_dropout)
        res_ols_dropout = sm.OLS(endog, exog).fit()
        res_vif_dropout.append(res_ols_dropout.rsquared)

    table = [[name, vif, dropout] for name, vif, dropout in zip(res_name, res_vif, res_vif_dropout)]
    headers = ["Feature", "VIF", "Dropout R2"]
    print(tabulate(table, headers, tablefmt="simple"))
    print('(OLS R2 with no dropout: %s)' % res_ols.rsquared)


def somersd(score, y_true, print_all=False):
    '''
    https://incipientanalyst.wordpress.com/2017/08/08/generic-python-code-for-classification-techniques/

    Calculate Somers D

    Parameters:
        scores - Probability of class 1.  Series or one-column Dataframe
        target - True target value (1 or 0).  Series or one-column Dataframe

    Returns:
        Prints statistics, return Somers D values
    '''
    target_name = y_true.name
    # Either Series or DataFrame, convert to DataFrame and name the target col
    score = pd.DataFrame(score)
    y_true = pd.DataFrame(y_true, columns=[target_name])

    TruthTable = pd.merge(y_true, score, how='inner', left_index=True, right_index=True)
    zeros = TruthTable[(TruthTable[target_name] == 0)].reset_index().drop(['index'], axis=1)
    ones = TruthTable[(TruthTable[target_name] == 1)].reset_index().drop(['index'], axis=1)
    from bisect import bisect_left, bisect_right
    zeros_list = sorted([zeros.iloc[j, 1] for j in zeros.index])
    zeros_length = len(zeros_list)
    disc = 0
    ties = 0
    conc = 0
    for i in ones.index:
        cur_conc = bisect_left(zeros_list, ones.iloc[i, 1])
        cur_ties = bisect_right(zeros_list, ones.iloc[i, 1]) - cur_conc
        conc += cur_conc
        ties += cur_ties
    pairs_tested = zeros_length * len(ones.index)
    disc = pairs_tested - conc - ties

    concordance = round(conc/pairs_tested, 2)
    discordance = round(disc/pairs_tested, 2)
    ties_perc = round(ties/pairs_tested, 2)
    Somers_D = (conc - disc)/pairs_tested

    if print_all:
        print('Pairs = ', pairs_tested)
        print('Conc =  ', conc)
        print('Disc =  ', disc)
        print('Tied =  ', ties)
        print('Concordance: %0.3f' % concordance)
        print('Discordance: %0.3f' % discordance)
        print('Tied:        %0.3f' % ties_perc)
        print('-------------------')
        print('Somers D:    %0.3f' % Somers_D)

    return Somers_D


def somersd_gini(score, target):
    '''
    https://qizeresearch.wordpress.com/2013/11/24/somers-d-calculation-r-and-python-code/
    Somer's D = (Num of Concordant Pair- Num of Disconcordant Pair) / Total number of pars (include tie)

    Based on the definition, it is computationally expensive to calculate somer's D.
    Mathematically, Somer's D is equal to Gini coefficient.
    Calculating Gini coefficient is much easier.

    Suppose we sort obs by their scores from low to high and divided them into 20 bins

        Gini [bin i] = (cum pct of event [i] + cum pct of event [i-1]) * (cum pct of no-event[i] - cum pct of no-event[i-1])

        Gini coefficient = 1 - sum(Gini[bin 1 to 20])

    Here, cum pct of event [i] = (num of obs with target=1 from bin 1 to i) / (total num of obs with target =1)
    '''
    score, target = (list(t) for t in zip(*sorted(zip(score, target))))

    ttl_num = len(score)
    bins = 20
    n = ttl_num/20

    sum_target = sum(target)
    sum_notarget = ttl_num-sum_target
    pct_target = []
    pct_notarget = []
    pct_target.append(0.0)
    pct_notarget.append(0.0)
    for i in range(1, bins):
        if i is not bins:
            # print(i, bins)
            bin_range = int((i*n-1))
            pct_target.append((sum(target[0:bin_range])+0.0)/sum_target)
            pct_notarget.append((i*n-sum(target[0:bin_range])+0.0)/sum_notarget)

    pct_target.append(1.0)
    pct_notarget.append(1.0)
    sd = []
    for i in range(1, bins+1):
        sd.append((pct_target[i]+pct_target[i-1])*(pct_notarget[i]-pct_notarget[i-1]))
        somersd = 1-sum(sd)

    return somersd


def ks(a, b):
    '''
    Perform Kolmogorov-Smirnov test on two Series.
    Plot overlaid histograms of the two distributuoins if the flag is True.

    Parameters:
        a - distribution 1
        b - distribution 2

    Returns
        KS test result as scipy.stats.stats.Ks_2sampResult
    '''
    ks = stats.ks_2samp(a, b)
    return ks


def woe_iv(X, y, event=1, print_on=True):
    '''
    Calculate woe of each feature category and information value.
    https://github.com/patrick201/information_value

    Parameters
        X - a pandas dataframe of explanatory features which should be discreted already
        y - 1-D numpy array target variable which should be binary
        event - value of binary stands for the event to predict
    Return
        Numpy array of woe dictionaries,
            each dictionary contains woe values for categories of each feature
        Numpy array of information value of each feature
    '''
    def feature_discretion(X):
        '''
        Discrete the continuous features of input data X, and keep other features unchanged.
        :param X : numpy array
        :return: the numpy array in which all continuous features are discreted
        '''
        temp = []
        for i in range(0, X.shape[-1]):
            x = X[:, i]
            x_type = type_of_target(x)
            if x_type == 'continuous':
                x1 = discrete(x)
                temp.append(x1)
            else:
                temp.append(x)
        return np.array(temp).T

    def discrete(x):
        '''
        Discrete the input 1-D numpy array using 5 equal percentiles
        :param x: 1-D numpy array
        :return: discreted 1-D numpy array
        '''
        res = np.array([0] * x.shape[-1], dtype=int)
        for i in range(5):
            point1 = stats.scoreatpercentile(x, i * 20)
            point2 = stats.scoreatpercentile(x, (i + 1) * 20)
            x1 = x[np.where((x >= point1) & (x <= point2))]
            mask = np.in1d(x, x1)
            res[mask] = (i + 1)
        return res

    check_target_binary(y)

    # Convert the X dataframe to 2d numpy array and capture the columns
    cols = X.columns
    X = X.as_matrix()

    X1 = feature_discretion(X)

    res_woe = []
    res_iv = []
    for i in range(0, X1.shape[-1]):
        x = X1[:, i]
        woe_dict, iv1 = woe_single_x(x, y, event)
        res_woe.append(woe_dict)
        res_iv.append(iv1)

    woe = np.array(res_woe)
    iv = np.array(res_iv)

    if print_on:
        table = [[col, json.dumps(w).replace('{', '').replace('}', '').replace(', ', '\n'), i]
                 for col, w, i in zip(cols, woe, iv)]
        headers = ["Feature", "WOE", "IV"]
        print(tabulate(table, headers, tablefmt="simple"))

    return woe, iv


def woe_single_x(x, y, event=1):
    '''
    calculate woe and information for a single featureself.

    Parameters:
        x - 1-D numpy starnds for single feature
        y - 1-D numpy array target variable
        event - value of binary stands for the event to predict
    Return:
        dictionary contains woe values for categories of this feature
        information value of this feature
    '''
    def count_binary(a, event=1):
        event_count = (a == event).sum()
        non_event_count = a.shape[-1] - event_count
        return event_count, non_event_count

    _WOE_MIN = -20
    _WOE_MAX = 20
    check_target_binary(y)

    event_total, non_event_total = count_binary(y, event=event)
    x_labels = np.unique(x)
    woe_dict = {}
    iv = 0
    for x1 in x_labels:
        y1 = y[np.where(x == x1)[0]]
        event_count, non_event_count = count_binary(y1, event=event)
        rate_event = 1.0 * event_count / event_total
        rate_non_event = 1.0 * non_event_count / non_event_total
        if rate_event == 0:
            woe1 = _WOE_MIN
        elif rate_non_event == 0:
            woe1 = _WOE_MAX
        else:
            woe1 = math.log(rate_event / rate_non_event)
        woe_dict[x1] = woe1
        iv += (rate_event - rate_non_event) * woe1
    return woe_dict, iv


def check_target_binary(y):
    '''
    Check if the target variable is binary, raise error if not.

    Parameters
        y - target series
    Return
        None
    '''
    y_type = type_of_target(y)
    if y_type not in ['binary']:
        raise ValueError('Label type must be binary')


def confusion_matrix_threshold(df, y_true_label, y_pred_label, threshold):
    '''
    Create a confusion matrix (tn, fp, fn, tp) at a given threshold.

    Parameters:
        df - data frame that holds the true and pred dataset
        y_true_label - column label of ground truth data, 1s and 0s
        y_pred_label - column label of predictions, probabilities
        threshold - the threshold to create that tn, fp, fn, tp values

    Returns:
        Confusuion matrix as numpy array
    '''
    tp = df[(df[y_true_label] == 1) & (df[y_pred_label] > threshold)].shape[0]
    fn = df[(df[y_true_label] == 1) & (df[y_pred_label] <= threshold)].shape[0]
    tn = df[(df[y_true_label] == 0) & (df[y_pred_label] < threshold)].shape[0]
    fp = df[(df[y_true_label] == 0) & (df[y_pred_label] >= threshold)].shape[0]

    cm = np.array([[tn, fp], [fn, tp]])
    return cm
