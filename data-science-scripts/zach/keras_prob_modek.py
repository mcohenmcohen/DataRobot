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
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.preprocessing import OneHotEncoder

# Other Notes
# https://blogs.rstudio.com/tensorflow/posts/2019-06-05-uncertainty-estimates-tfprobability/

# --------------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------------

games_season = pd.read_csv('https://s3.amazonaws.com/assets.datacamp.com/production/course_6554/datasets/games_season.csv')
games_tourney = pd.read_csv('https://s3.amazonaws.com/assets.datacamp.com/production/course_6554/datasets/games_tourney.csv')

SEASON = 2017
games_season = games_season.loc[games_season['season'] == SEASON]
games_tourney = games_tourney.loc[games_tourney['season'] == SEASON]

OHE = OneHotEncoder(categories='auto')
xvars = ['team_1', 'team_2', 'home']

X_train = OHE.fit_transform(games_season[xvars])
X_test = OHE.transform(games_tourney[xvars])

TARGET = ['score_1', 'score_2']
Y_train = games_season[TARGET]
Y_test = games_tourney[TARGET]

####################################################################################################
# Functions
####################################################################################################

# --------------------------------------------------------------------------------------------------
# Activations
# --------------------------------------------------------------------------------------------------

PI_INV = 1. / np.sqrt(np.pi)
NEG_HALF_LOG_2PI = -0.5*np.log(2*np.pi)
TF_NORM = tf.distributions.Normal(loc=0.0, scale=1.0)
SP_NORM = sp.stats.norm(loc=0.0, scale=1.0)


def probit_activation(x):
    normal = tf.distributions.Normal(loc=0.0, scale=1.0)
    return normal.cdf(x)


def diff_gauss_activation(x):
    mu = x[:,0] - x[:,1]
    sigma = x[:,2] - x[:,3]
    normal = tf.distributions.Normal(loc=0.0, scale=1.0)
    return normal.cdf(mu / sigma)


class NamedActivation(ks.layers.Activation):
    def __init__(self, activation_function, name, **kwargs):
        super(NamedActivation, self).__init__(activation_function, **kwargs)
        self.__name__ = name

get_custom_objects = ks.utils.generic_utils.get_custom_objects
get_custom_objects().update({'probit': NamedActivation(probit_activation, 'probit')})
get_custom_objects().update({'diff_gauss': NamedActivation(probit_activation, 'diff_gauss')})

# --------------------------------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------------------------------

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

CRPS_loss_sp = partial(CRPS_loss, pdf=SP_NORM.pdf, cdf=SP_NORM.cdf, mean=np.mean)


# prob of norm(0,1) > norm(0, 1)
# https://math.stackexchange.com/questions/1376005/cdf-of-the-difference-of-two-gaussian-mixtures
def prob_diff(a, b):
    return SP_NORM.cdf((a - b)/(np.std(a) + np.std(b)))


def  quick_ll(act, pred):
    logloss = -(act * np.log(pred) + (1 - act) * np.log(1 - pred))
    return logloss.mean()


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
    log2pi = n_dims*NEG_HALF_LOG_2PI

    log_likelihood = mse+sigma_trace+log2pi

    return K.mean(-log_likelihood)


def diff_gaussian_loss(ytrue, ypreds):

        n_dims = int(int(ypreds.shape[1])/2)
        mu_1 = ypreds[:, 0]
        mu_2 = ypreds[:, 1]

        sigma_1 = K.exp(ypreds[:, 2])
        sigma_2 = K.exp(ypreds[:, 3])
        sigma_sum = (sigma_1 + sigma_2)

        pred = TF_NORM.cdf((mu_1 - mu_2)/sigma_sum)
        act = TF_NORM.cdf((ytrue[: ,0] - ytrue[:, 1])/sigma_sum)

        logloss = -(act * K.log(pred) + (1 - act) * K.log(1 - pred))

        return K.mean(logloss)


####################################################################################################
# Keras model 1
####################################################################################################

Y_train_prob = prob_diff(games_season['score_1'], games_season['score_2'])
Y_test_prob = prob_diff(games_tourney['score_1'], games_tourney['score_2'])

print(quick_ll(games_season['score_1']>games_season['score_2'], Y_train_prob))
print(quick_ll(games_tourney['score_1']>games_tourney['score_2'], Y_test_prob))

in_all = ks.Input(shape=(X_train.shape[1],), dtype='float32', sparse=True, name='In-All')

out_mean_1 = ks.layers.Dense(1, kernel_initializer='zeros', bias_initializer='ones', activation='linear', name='mean-1')(in_all)
out_sd_1 = ks.layers.Dense(1, kernel_initializer='zeros', bias_initializer='zeros', activation='softplus', name='sd-1')(in_all)

out_mean_2 = ks.layers.Dense(1, kernel_initializer='zeros', bias_initializer='ones', activation='linear', name='mean-2')(in_all)
out_sd_2 = ks.layers.Dense(1, kernel_initializer='zeros', bias_initializer='zeros', activation='softplus', name='sd-2')(in_all)

out_mix_gauss = ks.layers.Concatenate()([out_mean_1, out_mean_2, out_sd_1, out_sd_2])

out_diff_mean = ks.layers.Subtract()([out_mean_1, out_mean_2])
out_sum_sd = ks.layers.Add()([out_sd_1, out_sd_2])
out_prob = ks.layers.Lambda(lambda x: x[0] / x[1])([out_diff_mean, out_sum_sd])
out_prob = ks.layers.Activation('probit')(out_prob)

model = ks.Model(in_all, [out_mix_gauss, out_prob])
model.compile(loss=[gaussian_nll, 'binary_crossentropy'], loss_weights=[.1, 10], optimizer=ks.optimizers.Adam(lr=0.01))
#model.summary()

model.fit(X_train, [Y_train, Y_train_prob], epochs=10, batch_size=2)

y_pred = model.predict(X_test)

print('Keras Test LogLoss', quick_ll(Y_test_prob, y_pred[1]))
print('Keras Test LogLoss', quick_ll(games_tourney['score_1']>games_tourney['score_2'], y_pred[1]))

####################################################################################################
# Keras model 2
####################################################################################################

out_sd = ks.layers.Dense(1, kernel_initializer='zeros', bias_initializer='zeros', activation='softplus', name='out-sd')(model_in)
out_sd = ks.layers.Concatenate(name='cat-sd')([out, out_sd])
model_sd = ks.Model(model_in, out_sd)
model_sd.compile(loss=gaussian_nll, optimizer=ks.optimizers.Adam(lr=0.001))
model_sd.fit(X_train, Y_train, epochs=5, batch_size=1)

model_sd_pred = model_sd.predict(X_test)
print('Keras CRPS (RMSE): ', np.sqrt(mean_squared_error(model_sd_pred[:,0], Y_test)))
print('Keras CRPS (CRPS): ', CRPS_loss_sp(Y_test, model_sd_pred))
np.round(model_sd_pred, 2)

####################################################################################################
# Compare
####################################################################################################

print('NGBoost Test MSE', mean_squared_error(y_preds_ngb, Y_test))
print('Keras MSE Test MSE', mean_squared_error(y_preds_keras_mse, Y_test))
print('Keras CRPS Test MSE: ', mean_squared_error(model_sd_pred[:,0], Y_test))
