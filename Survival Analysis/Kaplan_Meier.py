import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # for custom legends
import seaborn as sns
from lifelines import KaplanMeierFitter  # survival analysis library
from lifelines.statistics import logrank_test  # survival statistical testing
from IPython.display import Image
from IPython.core.display import HTML


def display_all(df):
    with pd.option_context("display.max_rows", 1000):
        with pd.option_context("display.max_columns", 1000):
            display(df)


def kaplan_meier(df, churn_var, tenure_var,
                 segmentation_var='', segmentation_label=['a', 'b'],
                 ylim_min=.5):
    '''
    From:  https://towardsdatascience.com/survival-analysis-in-python-a-model-for-customer-churn-e737c5242822

    For background, see Rich et al (2010), Singh et all (2011)

    - Estimates the “survival function” via the for a cohort of subjects
    - Does not provide an estimate of the magnitude of the difference in survival
      for the cohorts being compared
    - Each observation (case, customer) should have one "birth" (activation) and
      one "death" (churn)

    Parameters:
        churn_var - observation churn or not, 1 or 0
        duration_var - observation tenure or duration, the time since observing
        segmentation_var - The feature to segement the popultion, so that we can plot and see
                           the difference in curves between value of the given feature
    '''
    kmf = KaplanMeierFitter()
    T = df[tenure_var]  # duration
    C = df[churn_var]  # censorship - 1 if death/churn is seen, 0 if censored
    S = df[segmentation_var]

    palette = ["windows blue", "amber"]
    sns.set_palette(sns.xkcd_palette(palette))

    # SET UP PLOT
    ax = plt.subplot(111)
    plt.title('Kaplan-Meier Estimate of Driver Retention by Multiple Lines')
    sns.set_context("talk")

    d = {} #to store the models
    vlines = []
    i = 0

    # PLOT FITTED GRAPH
    # loop through segmentation variable, plot on same axes
    for segment in S.unique():
        ix = S == segment
        d['kmf{}'.format(i+1)] = kmf.fit(T.loc[ix], C.loc[ix], label=segment)
        ax = kmf.plot(ax=ax, figsize=(12, 6))

        ax.set_xlim([T.min(), T.max()])
        ax.set_ylim([ylim_min, 1])

        y_ = kmf.survival_function_[kmf.survival_function_.round(2) == .75].dropna().index[0]
        ymax_ = kmf.survival_function_[kmf.survival_function_.round(2) == .75].dropna()[i][y_]

        vlines.append([y_, ymax_])
        i += 1

    # PLOT ANNOTATION
    # for each intercept, plot a horizontal and a vertical line up to the fitted curve
    xmin_ = 0
    for i, xy in enumerate(vlines):
        xmax_ = xy[0]
        color = "xkcd:{}".format(palette[i])

        plt.axvline(x=xy[0], ymax=.5, alpha=.8, color=color, linestyle='dotted')  # axes fractional
        plt.hlines(y=.75, xmin=xmin_, xmax=xmax_, alpha=.8, color=color, linestyle='dotted')  # axes data coordinates
        xmin_ += xmax_  # to avoid overlapping hlines

    # position text label
    difference = vlines[1][0]-vlines[0][0]
    label_frac_x = (vlines[0][0]+(difference)/2)/T.max()-.07  # midpoint (minus .07 centering factor)
    label_frac_y = 0.2

    # label first line
    ax.annotate('Difference: {}'.format(difference),
                xy=(vlines[0][0], .62), xycoords='data', color='white',
                xytext=(label_frac_x, label_frac_y), textcoords='axes fraction',
                arrowprops=dict(arrowstyle="-|>",  # arrows removed for reability
                                fc="k", ec="k")
                )

    # label second line
    ax.annotate('Difference: {} '.format(difference),
                xy=(vlines[1][0], .62), xycoords='data', color='black',
                xytext=(label_frac_x, label_frac_y), textcoords='axes fraction',
                arrowprops=dict(arrowstyle="-|>",  # arrows removed for reability
                                fc="k", ec='k')
                )

    # LEGEND
    # override default legend
    patches = [mpatches.Patch(color="xkcd:windows blue", label=segmentation_label[0]),
               mpatches.Patch(color="xkcd:amber", label=segmentation_label[1])
               ]
    plt.legend(handles=[patches[0], patches[1]], title="User Segmentation", loc='best')
