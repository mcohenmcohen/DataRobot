import keras as ks
import keras.backend as K
import numpy as np
import pandas as pd
import scipy as sp
import tensorflow as tf

from functools import partial
from ngboost import NGBRegressor

from scipy.sparse import rand
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

# Other Notes
# https://blogs.rstudio.com/tensorflow/posts/2019-06-05-uncertainty-estimates-tfprobability/

# --------------------------------------------------------------------------------------------------
# CRPS
# --------------------------------------------------------------------------------------------------

PI_INV = 1. / np.sqrt(np.pi)
TF_NORM = tf.distributions.Normal(loc=0.0, scale=1.0)
SP_NORM = sp.stats.norm(loc=0.0, scale=1.0)

def CRPS_loss(y_true, y_pred, pdf=TF_NORM.prob, cdf=TF_NORM.cdf, mean=K.mean):
    """
    use pdf=TF_NORM.prob, cdf=TF_NORM.cdf, mean=K.mean
    or pdf=SP_NORM.pdf, cdf=SP_NORM, mean=np.mean

    https://arxiv.org/pdf/1910.03225.pdf
    https://github.com/TheClimateCorporation/properscoring/blob/master/properscoring/_crps.py
    https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf - equation 21
    """

    mu = y_pred[:,0]
    sig = y_pred[:,1]

    # standadized actuals
    std_y_true = (y_true - mu) / sig

    # CRPS
    crps = sig * (std_y_true * (2 * cdf(std_y_true) - 1) + 2 * pdf(std_y_true) - PI_INV)
    return mean(crps)

CRPS_loss_tf = partial(CRPS_loss, pdf=TF_NORM.prob, cdf=TF_NORM.cdf, mean=K.mean)
CRPS_loss_sp = partial(CRPS_loss, pdf=SP_NORM.pdf, cdf=SP_NORM.cdf, mean=np.mean)

def neg_log_lik_gaussian(y_true, y_pred):
    """https://github.com/keras-team/keras/issues/5650"""
    dist = tf.distributions.Normal(loc=y_pred[:,0], scale=y_pred[:,0])
    return -1.0 * dist.log_prob(y_true)

def simple_nll_gaussian(y_true, y_pred):
    """https://fairyonice.github.io/Create-a-neural-net-with-a-negative-log-likelihood-as-a-loss.html"""
    pred_mu = y_pred[:, 0]
    pred_sd = y_pred[:, 1]
    SE = (pred_mu - y_true) ** 2
    ms = SE/pred_sd + tf.log(pred_sd)
    return tf.reduce_mean(ms)


def gaussian_nll(ytrue, ypreds):
    """Keras implmementation og multivariate Gaussian negative loglikelihood loss function.
    This implementation implies diagonal covariance matrix.

    https://gist.github.com/sergeyprokudin/4a50bf9b75e0559c1fcd2cae860b879e

    Parameters
    ----------
    ytrue: tf.tensor of shape [n_samples, n_dims]
        ground truth values
    ypreds: tf.tensor of shape [n_samples, n_dims*2]
        predicted mu and logsigma values (e.g. by your neural network)

    Returns
    -------
    neg_log_likelihood: float
        negative loglikelihood averaged over samples

    This loss can then be used as a target loss for any keras model, e.g.:
        model.compile(loss=gaussian_nll, optimizer='Adam')

    """

    n_dims = int(int(ypreds.shape[1])/2)
    mu = ypreds[:, 0:n_dims]
    logsigma = ypreds[:, n_dims:]

    mse = -0.5*K.sum(K.square((ytrue-mu)/K.exp(logsigma)),axis=1)
    sigma_trace = -K.sum(logsigma, axis=1)
    log2pi = -0.5*n_dims*np.log(2*np.pi)

    log_likelihood = mse+sigma_trace+log2pi

    return K.mean(-log_likelihood)

# --------------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------------

# Boston
#X, Y = load_boston(True)

# Toy
# https://github.com/stanfordmlgroup/ngboost/blob/master/figures/toy.py

def gen_data(n=50, bound=1, deg=3, beta=1, noise=0.9, intcpt=-1):
    x = np.linspace(-bound, bound, n)[:, np.newaxis]
    h = np.linspace(-bound, bound, n)[:, np.newaxis]
    e = np.random.randn(*x.shape) * (0.1 + 10 * np.abs(x))
    y = 50 * (x ** deg) + h * beta + noise * e + intcpt
    return x, y.squeeze(), np.c_[h, np.ones_like(h)]

#X, Y, _ = gen_data(n=10000)

#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

games_season = pd.read_csv('https://s3.amazonaws.com/assets.datacamp.com/production/course_6554/datasets/games_season.csv')
games_tourney = pd.read_csv('https://s3.amazonaws.com/assets.datacamp.com/production/course_6554/datasets/games_tourney.csv')

SEASON = 2017
games_season = games_season.loc[games_season['season'] == SEASON]
games_tourney = games_tourney.loc[games_tourney['season'] == SEASON]

OHE = OneHotEncoder(categories='auto')
xvars = ['team_1', 'team_2', 'home']

OHE.fit_transform(games_season['team_1'])


X_train_1 = OHE.fit_transform(games_season['team_1'])
X_train_2 = OHE.transform(games_season['team_2'])

X_test_1 = OHE.transform(games_tourney['team_1'])
X_test_2 = OHE.transform(games_tourney['team_2'])

X_train_diff = X_train_1 - X_train_2
X_train_sum = X_train_1 + X_train_2

X_test_diff = X_test_1 - X_test_2
X_test_sum = X_test_1 + X_test_2

# prob of norm(0,1) > norm(0, 1)
# https://math.stackexchange.com/questions/1376005/cdf-of-the-difference-of-two-gaussian-mixtures
def prob_diff(a, b):
    return SP_NORM.cdf((a - b)/(np.std(a) + np.std(b)))

Y_train = prob_diff(games_season['score_1'], games_season['score_2'])
Y_test = prob_diff(games_tourney['score_1'], games_tourney['score_2'])

def probit_activation(x):
    normal = tf.distributions.Normal(loc=0.0, scale=1.0)
    return normal.cdf(x)


# --------------------------------------------------------------------------------------------------
# Ngboost
# --------------------------------------------------------------------------------------------------

ngb = NGBRegressor().fit(X_train, Y_train)
y_preds_ngb = ngb.predict(X_test)
print('NGBoost Test RMSE', np.sqrt(mean_squared_error(y_preds_ngb, Y_test)))

# --------------------------------------------------------------------------------------------------
# Keras MSE
# --------------------------------------------------------------------------------------------------

model_in = ks.Input(shape=(X_train.shape[1],), dtype='float32', sparse=False, name='In')
out = ks.layers.Dense(1, name='passthrough')(model_in)
model_1 = ks.Model(model_in, out)
model_1.compile(loss='mean_squared_error', optimizer=ks.optimizers.Adam(lr=0.01))
model_1.fit(X_train, Y_train, epochs=10, batch_size=1)

y_preds_keras_mse = model_1.predict(X_test)
print('Keras MSE Test MSE', np.sqrt(mean_squared_error(y_preds_keras_mse, Y_test)))

# --------------------------------------------------------------------------------------------------
# Keras CRPS
# --------------------------------------------------------------------------------------------------

out_sd = ks.layers.Dense(1, kernel_initializer='zeros', bias_initializer='zeros', activation='softplus', name='out-sd')(model_in)
out_sd = ks.layers.Concatenate(name='cat-sd')([out, out_sd])
model_sd = ks.Model(model_in, out_sd)
model_sd.compile(loss=gaussian_nll, optimizer=ks.optimizers.Adam(lr=0.001))
model_sd.fit(X_train, Y_train, epochs=5, batch_size=1)

model_sd_pred = model_sd.predict(X_test)
print('Keras CRPS (RMSE): ', np.sqrt(mean_squared_error(model_sd_pred[:,0], Y_test)))
print('Keras CRPS (CRPS): ', CRPS_loss_sp(Y_test, model_sd_pred))
np.round(model_sd_pred, 2)

# --------------------------------------------------------------------------------------------------
# Compare
# --------------------------------------------------------------------------------------------------

print('NGBoost Test MSE', mean_squared_error(y_preds_ngb, Y_test))
print('Keras MSE Test MSE', mean_squared_error(y_preds_keras_mse, Y_test))
print('Keras CRPS Test MSE: ', mean_squared_error(model_sd_pred[:,0], Y_test))
