import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
import utils_stats


def is_likely_categorical(data, column, count_thresh=0.05):
    '''
    Using a value count percentage metric to assess whether data is categorical or not.

    *Per Jett in CFDS slack: "If the number of unique rows is less than 5% of the column size,
     or if there are fewer than 60 unique the column is classified as categorical."

    Parameters:
        data - pandas DataFrame
        column - name of a column data to assess
        count_thresh - percentage of total value count
    Returns
        True is likely categorical, False otherwise
    '''
    if 1.*data[column].nunique()/data[column].count() < count_thresh:  # or some other threshold
        return True
    else:
        return False

def dynamic_subplots(data, feature_set, target_name, plot_type, figsize=(15, 10), cols=3):
    '''
    Plot a grid of subplots with a variable number of subplots and column.  The rows are determined
    by the number of subplots and columns.

    Parameters
        data - pandas dataframe
        plot_type - one of:  hist, box, scatter, residual

    '''
    tot = len(feature_set)
    # cols = 3
    rows = tot // cols
    if tot % cols != 0:
        rows += 1
    if plot_type == 'residual' or plot_type == 'residual_normality':
        rows = len(feature_set)
        tot = rows * 2
    # print(tot, cols, rows)

    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    col_idx = 0
    for i in range(rows):
        for j in range(cols):
            if plot_type == 'hist':
                axs[i, j].hist(data[feature_set[col_idx]].dropna(), bins=100)
                axs[i, j].set_title('')
                axs[i, j].set_xlabel(feature_set[col_idx])
            elif plot_type == 'scatter':
                x = data[feature_set[col_idx]]
                y = data[target_name]
                axs[i, j].scatter(x, y)  #, s=area, c=colors, alpha=0.5)
                axs[i, j].set_title('')
                axs[i, j].set_ylabel(target_name)
                axs[i, j].set_xlabel(feature_set[col_idx])
            elif plot_type == 'box':
                data[[feature_set[col_idx], target_name]].boxplot(by=feature_set[col_idx], ax=axs[i, j])
                fig = axs[i, j].get_figure()
                fig.suptitle('')
                axs[i, j].set_title('')
                axs[i, j].set_ylabel(target_name)
                axs[i, j].set_xlabel(feature_set[col_idx])
            elif plot_type == 'residual':
                feat = feature_set[i]
                X = data[feat]
                y = data[target_name]
                # X = sm.add_constant(X)
                ols_model = sm.OLS(y, X).fit()
                res = ols_model.resid

                if j == 0:
                    # plot residual with OLS
                    sm.graphics.plot_ccpr(ols_model, feat, ax=axs[i, j])
                    axs[i, j].set_title('')
                    axs[i, j].set_ylabel(target_name)
                    axs[i, j].set_xlabel(feat)
                if j == 1:
                    # plot residual as boxplot
                    df_bx = data[feat].copy().to_frame()
                    df_bx['res'] = res
                    if is_likely_categorical(data, feat):
                        df_bx.boxplot(by=feat, ax=axs[i, j])
                    else:
                        axs[i, j].scatter(df_bx[feat], df_bx['res'])

                    fig = axs[i, j].get_figure()
                    fig.suptitle('')
                    # axs[i, j].set_title('Residuals as boxplot')
                    axs[i, j].set_title('')
                    axs[i, j].set_ylabel('Residual')
                    axs[i, j].set_xlabel(feat)
                    axs[i, j].axhline(linewidth=1, color='r', linestyle='dashed')
            elif plot_type == 'residual_serial':
                # sns.residplot(data[feature_set[col_idx]], data[target_name], ax=axs[i, j], scatter_kws={"s": 6})
                feat = feature_set[col_idx]
                X = data[feat]
                y = data[target_name]
                # X = sm.add_constant(X)
                ols_model = sm.OLS(y, X).fit()
                res = ols_model.resid
                sm.graphics.plot_ccpr(ols_model, feat, ax=axs[i, j])
                axs[i, j].set_title('')
                # axs[i, j].set_ylabel(target_name)
                axs[i, j].set_xlabel(feat)
            elif plot_type == 'residual_normality':
                feat = feature_set[i]
                X = data[feat]
                X = sm.add_constant(X)
                y = data[target_name]
                ols_model = sm.OLS(y, X).fit()
                res = ols_model.resid

                if j == 0:
                    sm.qqplot(res, dist=stats.norm, distargs=(), a=0, loc=0, scale=1,
                              fit=True, line='45', ax=axs[i, j])
                if j == 1:
                    res.hist(bins=100, ax=axs[i, j])
                    axs[i, j].set_ylabel('Residual Histogram')
                # axs[i, j].set_ylabel('QQ Plot)
                axs[i, j].set_xlabel(feat)
            elif plot_type == 'het_sked':
                feat = feature_set[col_idx]
                X = data[feat]
                y = data[target_name]
                # X = sm.add_constant(X)
                ols_model = sm.OLS(y, X).fit()
                res = ols_model.resid
                df_bx = data[feat].copy().to_frame()
                df_bx['res'] = res
                df_bx['yhat'] = ols_model.fittedvalues

                # df_bx.plot(kind='scatter', x='yhat', y='res', ax=ax[i])
                # print(feat)
                # print(red.head())
                axs[i, j].scatter(df_bx['yhat'], df_bx['res'])
                axs[i, j].axhline(linewidth=1, color='r', linestyle='dashed')
                # plt.title('Univariate At Model Level Analysis Of Heteroscedasticity')
                axs[i, j].set_ylabel('{} OLS Residuals'.format(feat))
                axs[i, j].set_xlabel('OLS Fitted Predictions({})'.format(target_name))

            # End if
            col_idx += 1
            if col_idx >= tot:
                break
    return (fig, axs)


def plot_residuals_categorical(data, feature_set, target_name, figsize=(15, 5)):
    df = data
    for feat in feature_set:
        # do simple OLS per feature (cont and cat)
        # feat = 'GarageCars'
        X = df[feat]
        y = df[target_name]
        # X = sm.add_constant(X)
        ols_model = sm.OLS(y, X).fit()
    #     print(ols_model.summary())

        res = ols_model.resid

        # plot residual
        fig, axs = plt.subplots(1, 2, figsize=figsize)
        # fig = sm.graphics.plot_partregress_grid(ols_model, fig=fig)
        fig = sm.graphics.plot_ccpr(ols_model, feat, ax=axs[0])
        axs[0].set_title('Residuals with OLS')

        df_bx = df[feat].copy().to_frame()
        df_bx['res'] = res
        df_bx.boxplot(by=feat, ax=axs[1])
        fig = axs[1].get_figure()
        fig.suptitle('')
        axs[1].set_title('Residuals as boxplot')
        axs[1].axhline(linewidth=1, color='r', linestyle='dashed')
        # plt.boxplot(df[feat], res)


def plot_roc(y_true, y_pred):
    '''
    Plot the ROC curve for a given target and prediction data sets.
    '''
    auc = roc_auc_score(y_true, y_pred)

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    plt.figure(figsize=(6, 6))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw) #  label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Prediction dataset ROC')
    plt.legend(loc='lower right', title='AUC:  {}'.format(round(auc, 4)), handles=[])
    plt.show()


def plot_roc_dr(model, datasource, predictions,
                       ks_label=True, somersd_label=True, f1_label=True,
                       accuracy_label=True):
    '''
    Plot the ROC curve for a DataRobot model and dataset source

    Parameters:
        model - DataRobot model
        datasource - validation, crossvalidation, holdout
            eg, datarobot.enums.CHART_DATA_SOURCE.VALIDATION
        predictions - data frame that holds the true and pred dataset
    '''
    roc = model.get_roc_curve(datasource)

    auc = model.metrics.get('AUC').get(datasource)
    ks = model.metrics.get('Kolmogorov-Smirnov').get(datasource)
    f1 = roc.get_best_f1_threshold()
    title = 'AUC:  {}'.format(round(auc, 4))
    if ks_label:
        title += '\nKS:  {}'.format(round(ks, 4))
    if f1_label:
        title += '\nBest F1 threshold:  {}'.format(round(f1, 4))
    accuracy_label

    # Somers' D
    y_true = predictions['y_true']
    y_pred = predictions['class_1.0']
    sd = utils_stats.somersd(score=y_pred, y_true=y_true)
    if somersd_label:
        title += '\nSomers\' D:  {}'.format(round(sd, 4))

    # Accuracy
    # TODO:  Standard calculation of accuracy may differ from DR stacked predictions
    #        using the threshold from the DR best F1
    cm = utils_stats.confusion_matrix_threshold(predictions,
                                                'y_true', 'class_1.0', threshold=f1)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / predictions.shape[0]
    if accuracy_label:
        title += '\nAccuracy:  {}'.format(round(accuracy, 4))

    fp_max = max([point['false_positive_score'] for point in roc.roc_points])
    tp_max = max([point['true_positive_score'] for point in roc.roc_points])
    x = []
    y = []
    for point in roc.roc_points:
        # Normalize the fp and tp to plot on range 0 to 1
        fp = point.get('false_positive_score') / fp_max
        tp = point.get('true_positive_score') / tp_max
        x.append(fp)
        y.append(tp)

    plt.figure(figsize=(6, 6))
    plt.plot(x, y)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{} dataset ROC'.format(datasource).title())
    plt.legend(loc="lower right", title=title, handles=[])


def plot_lift_chart_dr(model, datasource, num_bins):
    '''
    Plot the lift chart for a DataRobot model and dataset source

    Parameters:
        model - DataRobot model
        datasource - validation, crossvalidation, holdout
            eg, datarobot.enums.CHART_DATA_SOURCE.VALIDATION
        num_bins - number of boins to plot (ideally divisible by 60,
                   otherwise its rounded down)
    '''
    lc = model.get_lift_chart(datasource)

    bins_df = pd.DataFrame(lc.bins[::int(60/num_bins)])
    bins_df.head()

    x = bins_df.index+1
    xi = [i for i in range(1, len(x)+1)]
    bins_df["X axis"] = x

    plt.figure(figsize=(20, 8))
    plt.plot(bins_df['X axis'], bins_df['predicted'], color='blue')
    plt.plot(bins_df['X axis'], bins_df['actual'], color='orange')
    plt.xticks(xi, x)
    plt.xlabel('Bins based on predicted value')
    plt.ylabel('Average target value')
    plt.show()


def plot_ks(distr_1, distr_2, bins=100, alpha=0.7, label_1='Pred True', label_2='Pred False'):
    plt.hist(distr_1, bins=bins, alpha=alpha, label=label_1)
    plt.hist(distr_2, bins=bins, alpha=alpha, label=label_2)
    plt.legend()
    plt.xlabel('Class 1 Prediction Probability')
    plt.title('Kolmogorov-Smirnov')
    plt.show()


def output_playbook_classification_accuracy(df_preds, cm_thresh=.5):
    '''
    This is a convenience function to output and plot metrics and values related to
    classification models in a notebook.

    Parameters:
        df_preds - A dataframe with classification target and prediction values.
            'y_true' - the target, 1s and 0s
            'class_1.0' - class 1 probabilities
            'class_0.0' - class 0 probabilities
        cm_thresh - The threshold for the confusion matrix

    Returns:

    '''
    y_true = df_preds['y_true']
    y_pred_1 = df_preds['class_1.0']
    y_pred_0 = df_preds['class_0.0']

    threshold = cm_thresh

    # Confusion matrix, accuracy
    print('Confusion matrix:')
    cm = utils_stats.confusion_matrix_threshold(df_preds, 'y_true', 'class_1.0', threshold)
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / df_preds.shape[0]
    print('Accuracy: %0.3f (%s/%s)' % (accuracy, (tp + tn), df_preds.shape[0]))

    # Somers' D:
    somersd = utils_stats.somersd(score=y_pred_1, y_true=y_true)
    print('Somers\' D: %0.3f' % somersd)

    # KS
    ks = utils_stats.ks(y_pred_1, y_pred_0)
    print('KS: statistic=%s, pvalue=%s' % (ks.statistic, ks.pvalue))

    # Plot the ROC curve
    plot_roc(y_true, y_pred_1)

    # Plot the KS histograms
    plot_ks(y_pred_1, y_pred_0)
