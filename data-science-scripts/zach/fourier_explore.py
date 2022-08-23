import itertools
import string
import numpy as np
import pandas as pd
import numpy.testing as npt
import tesla
import scipy.sparse as sparse
import skflow
from sklearn import datasets
from sklearn import metrics
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from functools import partial
from skflow.models import elasticnet_linear_regression

import skflow

module_rng = np.random.RandomState(222)

# Make some data - simulated
def periodic_with_trend(x, breakpt=250):
    n = x.shape[0]
    noise = np.random.normal(size=n).reshape(-1, 1) / 100
    mask = np.where(x <= breakpt, 0, 1)
    nonlin = 0.5*(x-breakpt) * mask
    return 10*np.sin(0.5*x) + 10*np.cos(1*x) + 10*np.cos(1.5*x) + 0.3*x + nonlin + noise
X = np.linspace(0.0, 500, 10000).reshape(-1, 1)
Y = periodic_with_trend(X)
X2 = np.linspace(500, 1000, 10000).reshape(-1, 1)
Y2 = periodic_with_trend(X2)


# TODO: save as csv and upload to S3
if False:

    start = pd.to_datetime('1990-01-01')

    X_new = np.arange(X.shape[0])
    X2_new = np.arange(X2.shape[0])

    train =  pd.DataFrame({'x':start + pd.to_timedelta(X_new.flatten(), unit='D'), 'y':Y.flatten()})
    test =  pd.DataFrame({'x':start + pd.to_timedelta(X2_new.flatten(), unit='D'), 'y':Y2.flatten()})

    train.to_csv('~/datasets/sines_nonlin_trend_train.csv', index=False)
    test.to_csv('~/datasets/sines_nonlin_trend_test.csv', index=False)


sets = [
    [
    'https://s3.amazonaws.com/datarobot_public_datasets/time_series/monthly_earth_co2_train.csv',
    'https://s3.amazonaws.com/datarobot_public_datasets/time_series/monthly_earth_co2_test.csv'
    ],
    [
    'https://s3.amazonaws.com/datarobot_public_datasets/time_series/weekly_earth_co2_train.csv',
    'https://s3.amazonaws.com/datarobot_public_datasets/time_series/weekly_earth_co2_test.csv'
    ],
    [
    'https://s3.amazonaws.com/datarobot_public_datasets/time_series/fpp_antidiabetic_drugs_train.csv',
    'https://s3.amazonaws.com/datarobot_public_datasets/time_series/fpp_antidiabetic_drugs_test.csv'
    ],
    [
    'https://s3.amazonaws.com/datarobot_public_datasets/time_series/wunderground_Chicago_actual_max_temp_train.csv',
    'https://s3.amazonaws.com/datarobot_public_datasets/time_series/wunderground_Chicago_actual_max_temp_test.csv'
    ],
    [
    'https://s3.amazonaws.com/datarobot_public_datasets/time_series/iso_ne_hourly_load_train.csv', # Bad
    'https://s3.amazonaws.com/datarobot_public_datasets/time_series/iso_ne_hourly_load_test.csv'
    ],
    [
    'https://s3.amazonaws.com/datarobot_public_datasets/time_series/internet_time_series_train.csv',
    'https://s3.amazonaws.com/datarobot_public_datasets/time_series/internet_time_series_test.csv'
    ],
    [
    'https://s3.amazonaws.com/datarobot_public_datasets/time_series/facebook_peyton_manning_train.csv',
    'https://s3.amazonaws.com/datarobot_public_datasets/time_series/facebook_peyton_manning_test.csv' # weird offset
    ],
    [
    'https://s3.amazonaws.com/datarobot_public_datasets/time_series/facebook_retail_train.csv',
    'https://s3.amazonaws.com/datarobot_public_datasets/time_series/facebook_retail_test.csv'
    ],
    [
    'https://s3.amazonaws.com/datarobot_public_datasets/time_series/hyndman_turkish_electricity_demand_train.csv',
    'https://s3.amazonaws.com/datarobot_public_datasets/time_series/hyndman_turkish_electricity_demand_test.csv'
    ],
]

for s in sets:

    print(s[0])

    # Make some data - Real
    train = pd.read_csv(s[0])
    test = pd.read_csv(s[1])

    X = np.arange(train.shape[0]).astype(np.float32).reshape(-1, 1)
    Y = train[['y']].values.astype(np.float32).reshape(-1, 1)
    X2 = X.max() + np.arange(test.shape[0]).astype(np.float32).reshape(-1, 1)
    Y2 = test[['y']].values.astype(np.float32).reshape(-1, 1)
    print(X.max())
    print(Y.min())
    print(Y.max())

    scale = True
    plots = False

    if scale:
        #scale_x = MinMaxScaler((0, 1000))
        scale_y = StandardScaler()
        #X = scale_x.fit_transform(X)
        Y = scale_y.fit_transform(Y)
        #X2 = scale_x.transform(X2)
        Y2 = scale_y.transform(Y2)

    # Fit a model
    dnn = skflow.models.neural_decomposition(
        elasticnet_linear_regression(l1=1),
        n_fourier_terms=64,
        #freq_regularizer=1,
        freq_init='he_uniform',  # he_uniform fourier_frequencies
        phase_init='he_uniform',  # he_uniform fourier_phases
        nonlin_activation='relu',  # softexp relu prelu
        nonlin_W_init = 'zero',  # zero, he_uniform
        n_hidden_nonlin_units=8)
    regressor = skflow.TensorFlowEstimator(
        dnn, n_classes=0, optimizer='L-BFGS-B',  # L-BFGS-B
        steps=50000, learning_rate=0.01,
        second_order_max_iter=1000000, batch_size=10000, use_basin_hopping=True,
        basinhopping_niter=200
        )
    regressor.fit(X, Y)

    # Todo: show max and min freq/offset

    # Predict
    P = regressor.predict(X)
    P2 = regressor.predict(X2)
    print('Scaled Scores')
    print(metrics.mean_squared_error(Y, P))
    print(metrics.mean_squared_error(Y2, P2))

    if scale:
        Y = scale_y.inverse_transform(Y)
        Y2 = scale_y.inverse_transform(Y2)
        P = scale_y.inverse_transform(P)
        P2 = scale_y.inverse_transform(P2)
        print('Raw Scores')
        print(metrics.mean_squared_error(Y, P))
        print(metrics.mean_squared_error(Y2, P2))

    if plots:
        plt.plot(X, Y)
        plt.plot(X, P)
        plt.show()
        plt.plot(X2, Y2)
        plt.plot(X2, P2)
        plt.show()
