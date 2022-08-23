# Source reference:
# https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python

import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn import preprocessing
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import time
import matplotlib
import matplotlib.pyplot as plt
import datarobot as dr

matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('- Time: %r %0.2f' % (method.__name__, (te - ts)))
        return result
    return timed


@timeit
def best_fit_distribution(data, bins=200, ax=None):
    '''
    Model data by finding best fit distribution to data.

    Parameters:
        data - feature as pd.Series
        bin - num histogram bins
        ax - axes to plot each distribution on

    Returns:
        (best distribution name, best distribution params)
    '''
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS = [
        st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
        st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
        st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
        st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
        st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
        st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,st.levy_stable,
        st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
        st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
        st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
        st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
    ]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                t1 = time.time()
                params = distribution.fit(data)
                # print('Time %.3f - %s distribution.fit' % ((time.time()-t1), distribution.name))

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass

    return (best_distribution.name, best_params)


def make_pdf(dist, params, size=10000, normalize=True):
    '''
    Generate distributions's probability distribution function.

    Parameters:
        dist - scipy distribtution object
        params - (a, loc, scale, etc..)
        size - for pdf x linspace

    Returns:
        pdf Series
    '''

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    if normalize == True:
        # Normalize pdf
        pdf_norm = pdf.reset_index().rename(index=str, columns={"index": "data", 0: "pdf"})
        x = pdf_norm.pdf.values.reshape(-1,1)
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        pdf_norm.pdf = x_scaled
        pdf_norm = pdf_norm.set_index('data')
        pdf = pdf_norm.pdf  # Dataframe to Series

    return pdf


# https://datarobot-public-api-client.readthedocs-hosted.com/en/v2.8.1/api/training_predictions.html
# https://datarobot-public-api-client.readthedocs-hosted.com/en/v2.8.1/entities/training_predictions.html
@timeit
def get_training_predictions(project_id, model_id):
    '''
    Retrieve the all original cross validation fold assignments to the records in the training set.
    If a model id is passed, then return the model's predictions of the training set.

    Parameters:
        project id - string
        model id - string

    Returns:
        DataFrame of the training set records as rows, with 2 columns:
            - the cross validation fold assignment
            - the predictions value (TODO: this is 0.0 for all recrods.  Need to investigate)
    '''
    model = dr.Model.get(project=project_id, model_id=model_id)

    try:
        training_predictions_job = model.request_training_predictions(dr.enums.DATA_SUBSET.ALL)
        training_predictions = training_predictions_job.get_result_when_complete()
        print('Training predictions job for {} are requested and ready'.format(training_predictions.prediction_id))
    except:  # throws exception if alredy requested
        print('Training predictions are already requested and are ready.')

        # Fetch all training predictions for a project
        # (dr.TrainingPredictions.list returns all model training predictions calculated
        #  in a project in chron order.  Index 0 is the most recent model and its predictions.)
        all_training_predictions = dr.TrainingPredictions.list(project_id)

        # Match the model id to one of the project's training prediction datasets
        # and get its prediction id.
        for i, preds in enumerate(all_training_predictions):
            if preds.model_id == model.id:
                prediction_id = (all_training_predictions[i].prediction_id)
                break

        all_training_predictions = dr.TrainingPredictions.list(project_id)
        prediction_id = (all_training_predictions[0].prediction_id)

        # Getting training predictions by id
        training_predictions = dr.TrainingPredictions.get(project_id, prediction_id)

    return training_predictions.get_all_as_dataframe()


# Looking to automate this for the top n features using the API
@timeit
def get_model_top_features(project_id, model_id, top_n):
    '''
    Get the top n features from the model feature impact.

    Parameters:
        project id
        model id
        top_n - int number of features

    Returns:
        List of top n features, in order of feature importance
    '''
    model = dr.Model.get(project=project_id, model_id=model_id)
    print('Model type:', model.model_type)

    for i in range(100):
        time.sleep(3)
        try:
            print('Requesting feture impact for model %s' % model_id)
            feat_impacts = model.request_feature_impact().get_result_when_complete(max_wait=100)  # should be already requested as part of API deploy
        except:
            print('Feature impact compute done.')
            break

    feat_impacts = model.get_feature_impact()

    # feat_impacts should already be an ordered list by most impactful (but if not):
    from operator import itemgetter
    feat_impacts = sorted(feat_impacts, key=itemgetter('impactNormalized'), reverse=True)

    top_n_features = []
    for i, feat_imp in enumerate(feat_impacts):
        top_n_features.append(feat_impacts[i]['featureName'])
        # print('*****%s: %s\n' % (i,feat_imp))

    return top_n_features[:top_n]


#def draw_proba_univariate_values(x, prob_thresh, n_vals):
@timeit
def sample_low_proba_univariate_values(x, prob_thresh):
    '''
    Draw values from feature x that are below a given probability threshold
    according to a kernel density estimate.
    - 1 from left tail and 1 from right tail?  or n_vals?

    Parameters:
        x  - univariate array of data - Series, np array, or list
        prob_thresh - minimum probability threshold
        (n_vals - number of low prob values to return)

    Return:
        List of values of length 2 (or n_vals?)
    '''
    # Convert x to Series to dropna
    x = pd.Series(x).dropna()

    # Generate a kde from the data, using the optimal bandwidth
    grid = GridSearchCV(KernelDensity(),
                        {'bandwidth': np.linspace(0.1, 1.0, 30)},
                        cv=20)  # 20-fold cross-validation
    grid.fit(x[:, None])
    print('grid best params:', grid.best_params_)
    kde = grid.best_estimator_

    # Evaluate the density model on the data across an x range grid
    # - 1000 should be granular enough
    # - could extend beyond x range?
    x_grid = np.linspace(x.min(), x.max(), 1000)

    # pdf values
    pdf = np.exp(kde.score_samples(x_grid[:, None]))
    # print('pdf:',pdf)

    fig, ax = plt.subplots()
    ax.plot(x_grid, pdf, linewidth=3, alpha=0.5, label='bw=%.2f' % kde.bandwidth)
    ax.hist(x, 30, fc='gray', histtype='stepfilled', alpha=0.3, normed=True)
    ax.legend(loc='upper left')
    ax.set_xlim(x.min(), x.max())

    # dataframe of x_grid and pdf values
    df = pd.DataFrame({'x_grid': x_grid,
                       'pdf': pdf
                       })

    # subset x below prob thresh data in the left tail
    l_tail = df[(df['pdf'] < prob_thresh) & (df['x_grid'] < np.median(x))]
    # subset x below prob thresh data in the left tail
    r_tail = df[(df['pdf'] < prob_thresh) & (df['x_grid'] > np.median(x))]
    # any local minima below prob thresh?
    # mid_low_prob?
    # tails = np.concatenate(l_tail, r_tail)

    # draw 2 (n_val) random samples, equal between tails and append to list
    samples = []  # np.array()
    i = 0
    n_val = 2  # nvals
    while i < n_val:
        r = [r_tail.sample(1).iloc[0]['x_grid']]
        l = [l_tail.sample(1).iloc[0]['x_grid']]
        samples = samples + r + l
        i += 2
    print('samples:', samples)

    samples = sorted(samples)
    return samples


def sample_low_proba_from_pdf(pdf, prob_thresh=0.1):
    '''
    Sample extreme values from the pdf

    Pull random samples from a pdf with values less than the threshold, from both tails
    (TODO: any other local minima below prob thresh other than the tails in the middle?)

    Parameters:
        pdf - pandas Series
        prob_thresh - threshold

    Returns:
        list of sampled values
    '''
    max_value = pdf.idxmax()

    pdf_subset = pdf[pdf < prob_thresh].dropna()
    pdf_subset = pdf_subset.reset_index()
    # import pdb; pdb.set_trace()
    # l_tail = pdf_subset['data'][(pdf_subset.data < max_value) & (pdf_subset.pdf < prob_thresh)]
    # r_tail = pdf_subset['data'][(pdf_subset.data > max_value) & (pdf_subset.pdf < prob_thresh)]
    l_tail = pdf_subset['data'][pdf_subset.data < max_value].values
    r_tail = pdf_subset['data'][pdf_subset.data > max_value].values

    # Draw 2 (n_val) random samples, equal between tails and append to list
    samples = []  # np.array()
    i = 0
    n_val = 2  # nvals
    import random
    while i < n_val:
        if l_tail.shape[0] > 0:
            l = [random.choice(l_tail)]
            samples = samples + l
        if r_tail.shape[0] > 0:
            r = [random.choice(r_tail)]
            samples = samples + r
        i += 2

    return sorted(samples)
